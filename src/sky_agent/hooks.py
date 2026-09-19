from dataclasses import dataclass
import hashlib
import json
import threading
from contextlib import contextmanager
from typing import Protocol

from .execution import ExecutionContext, ToolError
from .persistence import PersistenceError
from .tools import Tool


def conversation_snapshot(messages: list[dict]) -> str:
    """Bound evidence for hooks without sharing mutable history or reasoning fields."""
    indices = list(range(max(0, len(messages) - 8), len(messages)))
    first_user = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
    if first_user is not None and first_user not in indices:
        indices.insert(0, first_user)
    items = []
    truncated = len(indices) < len(messages)
    for index in indices:
        message = messages[index]
        role = message.get("role")
        if role not in {"user", "assistant", "tool"}:
            continue
        content = message.get("content") or ""
        if not isinstance(content, str):
            content = "[unsupported content omitted]"
        truncated |= len(content) > 2000
        item = {"role": role, "content": content[:2000]}
        if role == "tool":
            item["tool_call_id"] = str(message.get("tool_call_id", ""))[:200]
        items.append(item)
    return json.dumps({"messages": items, "truncated": truncated}, ensure_ascii=False)


@dataclass(frozen=True)
class ToolRequest:
    tool_name: str
    arguments_json: str
    conversation_json: str
    read_only: bool
    permission_category: str | None
    workspace: str | None

    @classmethod
    def create(cls, tool: Tool, arguments: dict, conversation_json: str):
        return cls(tool.name, json.dumps(arguments, sort_keys=True, ensure_ascii=False, allow_nan=False),
                   conversation_json, tool.read_only, tool.permission_category,
                   str(tool.workspace) if tool.workspace else None)

    @property
    def arguments(self) -> dict:
        return json.loads(self.arguments_json)

    @property
    def fingerprint(self) -> str:
        payload = self.tool_name + "\n" + (self.workspace or "") + "\n" + self.arguments_json
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class BeforeToolHook(Protocol):
    def before_tool(self, request: ToolRequest, context: ExecutionContext) -> "HookDecision | None":
        """Return normally to continue, or return/raise a decision to veto."""
        ...


@dataclass(frozen=True)
class HookDecision:
    """A hook result. ``allow`` is advisory and can never override policy."""

    decision: str
    reason: str = ""

    def __post_init__(self):
        if self.decision not in ("allow", "deny", "ask"):
            raise ValueError("Hook decision must be allow, deny, or ask")
        if not isinstance(self.reason, str) or len(self.reason) > 2000:
            raise ValueError("Hook decision reason must be a string of at most 2000 characters")


class HookManager:
    """Dispatch optional lifecycle callbacks without putting them in the turn loop."""

    def __init__(self, hooks=()):
        self.hooks = tuple(hooks)
        self._lock = threading.RLock()

    @contextmanager
    def _serialized(self, context, *, cleanup=False):
        while not self._lock.acquire(timeout=0.05):
            if not cleanup:
                context.check_cancelled()
        try:
            yield
        finally:
            self._lock.release()

    @staticmethod
    def _payload(kwargs):
        # Each callback gets its own JSON copy; private model reasoning stays in
        # the transcript needed by the provider, not in extension payloads.
        payload = json.loads(json.dumps(kwargs, ensure_ascii=False, allow_nan=False))
        for message in payload.get("messages", []):
            message.pop("reasoning_content", None)
        if isinstance(payload.get("response"), dict):
            payload["response"].pop("reasoning_content", None)
        return payload

    @staticmethod
    def _failed(context, hook, method, exc):
        context.emit("hook_failed", hook=type(hook).__name__, method=method,
                     error=type(exc).__name__)

    def gate(self, method, context, **kwargs):
        """Run a gate. Callback errors fail closed and cannot be swallowed."""
        with self._serialized(context):
            for hook in self.hooks:
                callback = getattr(hook, method, None)
                if callback is None:
                    continue
                context.check_cancelled()
                context.emit("hook_started", hook=type(hook).__name__, method=method)
                try:
                    result = callback(context, **self._payload(kwargs))
                    if result is not None:
                        raise TypeError("Lifecycle gates must return None or raise ToolError")
                except PersistenceError:
                    raise
                except ToolError as exc:
                    self._failed(context, hook, method, exc)
                    raise
                except Exception as exc:
                    self._failed(context, hook, method, exc)
                    raise ToolError("hook_failed", f"{method} hook failed") from exc
                context.emit("hook_finished", hook=type(hook).__name__, method=method)
                context.check_cancelled()

    def before_tool(self, request, context):
        decision = HookDecision("allow")
        with self._serialized(context):
            for hook in self.hooks:
                callback = getattr(hook, "before_tool", None)
                if callback is None:
                    continue
                context.check_cancelled()
                context.emit("hook_started", hook=type(hook).__name__, method="before_tool")
                try:
                    result = callback(request, context)
                    if result is not None and not isinstance(result, HookDecision):
                        raise TypeError("Before-tool hooks must return HookDecision or None")
                except PersistenceError:
                    raise
                except ToolError as exc:
                    self._failed(context, hook, "before_tool", exc)
                    raise
                except Exception as exc:
                    self._failed(context, hook, "before_tool", exc)
                    raise ToolError("hook_failed", "A before-tool hook failed; execution denied") from exc
                context.emit("hook_finished", hook=type(hook).__name__, method="before_tool",
                             decision=result.decision if result else None,
                             reason=result.reason if result else "", fingerprint=request.fingerprint)
                context.check_cancelled()
                if result and result.decision == "deny":
                    return result
                if result and result.decision == "ask":
                    decision = HookDecision("ask", (decision.reason + "\n" + result.reason).strip()[:2000])
        return decision

    def observers(self, method, context, *, cleanup=False, **kwargs):
        """Run all observers, recording failures while preserving the run result."""
        failures = []
        with self._serialized(context, cleanup=True):
            for hook in self.hooks:
                try:
                    callback = getattr(hook, method, None)
                    if callback is None:
                        continue
                    # Cleanup must still run if the journal is unavailable.
                    try:
                        context.emit("hook_started", hook=type(hook).__name__, method=method)
                    except BaseException as exc:
                        if not cleanup:
                            raise
                        failures.append(exc)
                    result = callback(context, **self._payload(kwargs))
                    if result is not None:
                        raise TypeError("Observer and cleanup hooks must return None")
                    context.emit("hook_finished", hook=type(hook).__name__, method=method)
                except BaseException as exc:
                    if not cleanup and (not isinstance(exc, Exception) or isinstance(exc, PersistenceError)):
                        raise
                    failures.append(exc)
                    try:
                        self._failed(context, hook, method, exc)
                    except BaseException as audit_error:
                        if not cleanup:
                            raise
                        failures.append(audit_error)
        return failures

    def stop(self, context, **kwargs):
        """Best-effort cleanup; every stop callback gets a chance to run."""
        return self.observers("stop", context, cleanup=True, **kwargs)
