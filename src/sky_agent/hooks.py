from dataclasses import dataclass
import hashlib
import json
from typing import Protocol

from .execution import ExecutionContext
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
    def before_tool(self, request: ToolRequest, context: ExecutionContext) -> None:
        """Return normally to continue; raise ToolError to veto the call."""
        ...
