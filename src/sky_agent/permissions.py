from dataclasses import dataclass, field
import threading
import time
from typing import Callable, Protocol

from .execution import ExecutionContext, ToolError
from .hooks import HookDecision, ToolRequest
from .persistence import PersistenceError

PERMISSION_MODES = ("allow", "default", "acceptEdits", "auto", "readOnly")


@dataclass(frozen=True)
class Classification:
    decision: str
    reason: str

    def __post_init__(self):
        if self.decision not in {"allow", "deny", "ask"}:
            raise ValueError("Invalid classifier decision")
        if not isinstance(self.reason, str) or not self.reason.strip() or len(self.reason) > 2000:
            raise ValueError("Classification reason must contain 1 to 2000 characters")


class ActionClassifier(Protocol):
    def classify(self, request: ToolRequest) -> Classification: ...


@dataclass(frozen=True)
class ApprovalRequest:
    call: ToolRequest
    reason: str
    source: str
    cancel: threading.Event


@dataclass
class PermissionPolicy:
    read_only: bool = False
    denied: set[str] = field(default_factory=set)
    ask: set[str] = field(default_factory=set)
    approve: Callable[[str, dict], bool] | None = None
    mode: str = "allow"
    allowed: set[str] = field(default_factory=set)
    classifier: ActionClassifier | None = None
    rejection_threshold: int = 3
    human_approve: Callable[[ApprovalRequest], bool] | None = None

    def __post_init__(self):
        if self.mode not in PERMISSION_MODES:
            raise ValueError(f"Unknown permission mode: {self.mode}")
        if type(self.rejection_threshold) is not int or self.rejection_threshold < 1:
            raise ValueError("rejection_threshold must be a positive integer")

    def new_session(self):
        return PermissionHook(self)


class PermissionHook:
    """Per-run permission state. The tool runtime owns this hook, not the loop."""

    def __init__(self, policy: PermissionPolicy):
        self.policy = policy
        self.consecutive_denials = 0
        self.human_fallback = False
        self._lock = threading.Lock()

    def _ask(self, request, context, source, reason):
        context.check_cancelled()
        context.emit("permission_requested", source=source, reason=reason,
                     fingerprint=request.fingerprint)
        context.check_cancelled()
        try:
            if self.policy.human_approve:
                approved = self.policy.human_approve(ApprovalRequest(request, reason, source, context.cancel)) is True
            elif self.policy.approve:
                approved = self.policy.approve(request.tool_name, request.arguments) is True
            else:
                return "deny", "approval_unavailable", "Human approval is required but no approver is configured"
        except (ToolError, PersistenceError):
            raise
        except Exception as exc:
            return "deny", "approval_error", f"Approval failed ({type(exc).__name__})"
        context.check_cancelled()
        return ("allow" if approved else "deny"), "human", (
            "Approved for this exact call" if approved else "Approval declined or unavailable")

    def _decide(self, request, context, hook_decision):
        policy = self.policy
        if request.tool_name in policy.denied:
            return "deny", "deny_rule", "Tool is explicitly denied"
        if (policy.read_only or policy.mode == "readOnly") and not request.read_only:
            return "deny", "read_only", "Read-only policy denies this tool"
        if hook_decision.decision == "deny":
            return "deny", "hook", hook_decision.reason or "Hook denied this tool"
        if request.tool_name in policy.ask or hook_decision.decision == "ask":
            source = "ask_rule" if request.tool_name in policy.ask else "hook"
            reason = "Explicit rule requires human approval" if source == "ask_rule" else "Hook requires human approval"
            if hook_decision.reason:
                reason += ": " + hook_decision.reason
            return self._ask(request, context, source, reason)
        if request.tool_name in policy.allowed:
            return "allow", "allow_rule", "Tool is explicitly allowed"
        if policy.mode == "allow":
            return "allow", "allow_mode", "Legacy allow mode"
        if policy.mode == "readOnly" or policy.read_only:
            return "allow", "read_only", "Tool is permitted by read-only policy"
        if policy.mode in {"acceptEdits", "auto"} and request.permission_category == "edit" and request.workspace:
            return "allow", "accept_edits", "Validated direct workspace edit"
        if request.permission_category == "read" and request.read_only:
            return "allow", "safe_read", "Trusted read-tool capability"
        if policy.mode != "auto":
            return self._ask(request, context, "mode", "Permission mode requires human approval")
        if self.human_fallback:
            return self._ask(request, context, "denial_limit", "Session switched to human approval after repeated classifier denials")
        if policy.classifier is None:
            return self._ask(request, context, "classifier_unavailable", "Auto mode has no classifier")
        started = time.monotonic()
        try:
            context.check_cancelled()
            result = policy.classifier.classify(request)
            if not isinstance(result, Classification):
                raise ValueError("Classifier did not return a Classification")
        except (ToolError, PersistenceError):
            raise
        except Exception as exc:
            context.emit("permission_classification", decision="error", error=type(exc).__name__,
                         fingerprint=request.fingerprint, duration=time.monotonic() - started)
            return self._ask(request, context, "classifier_error", "Classifier failed; human approval is required")
        context.check_cancelled()
        if result.decision == "deny":
            self.consecutive_denials += 1
            self.human_fallback = self.consecutive_denials >= policy.rejection_threshold
        else:
            self.consecutive_denials = 0
        context.emit("permission_classification", decision=result.decision, reason=result.reason,
                     fingerprint=request.fingerprint, duration=time.monotonic() - started,
                     consecutive_denials=self.consecutive_denials, human_fallback=self.human_fallback)
        if self.human_fallback:
            return self._ask(request, context, "denial_limit", result.reason)
        if result.decision == "ask":
            return self._ask(request, context, "classifier", result.reason)
        return result.decision, "classifier", result.reason

    def before_tool(self, request: ToolRequest, context: ExecutionContext,
                    hook_decision: HookDecision = HookDecision("allow")) -> None:
        started = time.monotonic()
        while not self._lock.acquire(timeout=0.05):
            context.check_cancelled()
        try:
            context.check_cancelled()
            decision, source, reason = self._decide(request, context, hook_decision)
            context.check_cancelled()
            context.emit("permission_decision", decision=decision, source=source, reason=reason,
                         fingerprint=request.fingerprint, duration=time.monotonic() - started,
                         mode=self.policy.mode, consecutive_denials=self.consecutive_denials,
                         human_fallback=self.human_fallback)
            context.check_cancelled()
            if decision != "allow":
                code = "hook_denied" if source == "hook" and hook_decision.decision == "deny" else "permission_denied"
                raise ToolError(code, reason, {"source": source})
        finally:
            self._lock.release()
