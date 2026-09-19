"""Versioned, session-local plans; the journal is the durable source of truth."""

import json
import re
import threading

from jsonschema import Draft202012Validator, ValidationError

from .execution import ToolError
from .persistence import PersistenceError

STATUSES = ("pending", "in_progress", "completed", "blocked", "cancelled")
TERMINAL = {"completed", "cancelled"}
ITEM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"},
        "content": {"type": "string", "minLength": 1, "maxLength": 500},
        "status": {"type": "string", "enum": list(STATUSES)},
        "reason": {"type": "string", "maxLength": 500},
    },
    "required": ["id", "content", "status"],
}
WRITE_PROPERTIES = {
    "expected_revision": {"type": "integer", "minimum": 0},
    "todos": {"type": "array", "maxItems": 50, "items": ITEM_SCHEMA},
    "change_reason": {"type": "string", "maxLength": 1000},
}
WRITE_REQUIRED = ["expected_revision", "todos"]
_validator = Draft202012Validator({
    "type": "object", "properties": WRITE_PROPERTIES,
    "required": WRITE_REQUIRED, "additionalProperties": False,
})


def validate_write(expected_revision, todos, change_reason=""):
    try:
        _validator.validate({"expected_revision": expected_revision, "todos": todos,
                             "change_reason": change_reason})
    except ValidationError as exc:
        raise ToolError("invalid_arguments", exc.message[:2000]) from exc
    if type(expected_revision) is not int:
        raise ToolError("invalid_arguments", "expected_revision must be an integer")
    identifiers = set()
    active = 0
    for item in todos:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", item["id"]):
            raise ToolError("invalid_arguments", "Invalid todo ID")
        if item["id"] in identifiers:
            raise ToolError("invalid_arguments", "Todo IDs must be unique")
        identifiers.add(item["id"])
        if not item["content"].strip():
            raise ToolError("invalid_arguments", "Todo content must not be blank")
        if item["status"] in {"blocked", "cancelled"} and not item.get("reason", "").strip():
            raise ToolError("invalid_arguments", "Blocked/cancelled items require a reason")
        active += item["status"] == "in_progress"
    if active > 1:
        raise ToolError("invalid_arguments", "At most one todo may be in_progress")


def next_snapshot(current, expected_revision, todos, change_reason=""):
    validate_write(expected_revision, todos, change_reason)
    if expected_revision != current["revision"]:
        raise ToolError("todo_conflict", "Plan changed; read it again before writing",
                        {"expected_revision": expected_revision, "current_revision": current["revision"]})
    incoming = {item["id"]: item for item in todos}
    for old in current["todos"]:
        new = incoming.get(old["id"])
        removed_unfinished = new is None and old["status"] not in TERMINAL
        reopened = new is not None and old["status"] in TERMINAL and new["status"] not in TERMINAL
        if (removed_unfinished or reopened) and not change_reason.strip():
            raise ToolError("todo_reason_required", "Removing unfinished or reopening terminal items requires change_reason")
    return {"revision": current["revision"] + 1,
            "todos": json.loads(json.dumps(todos, ensure_ascii=False))}


def replay_update(current, event):
    """Validate historical writes instead of trusting the last JSON object."""
    snapshot = next_snapshot(current, event["expected_revision"], event["todos"], event["change_reason"])
    if type(event["revision"]) is not int or event["revision"] != snapshot["revision"]:
        raise ValueError("Invalid todo revision sequence")
    return snapshot


def summarize(snapshot):
    return {"revision": snapshot["revision"], "total": len(snapshot["todos"]),
            "counts": {status: sum(t["status"] == status for t in snapshot["todos"]) for status in STATUSES},
            "unfinished_ids": [t["id"] for t in snapshot["todos"] if t["status"] not in TERMINAL]}


class TodoStore:
    def __init__(self, store):
        self.store = store
        self._lock = threading.Lock()
        self._snapshot = {"revision": 0, "todos": []}

    def read(self):
        with self._lock:
            return json.loads(json.dumps(self._snapshot, ensure_ascii=False))

    def write(self, context, expected_revision, todos, change_reason=""):
        # Match event dispatch lock ordering. Release the state lock before any
        # observer runs, so observers may inspect the committed plan.
        with context.event_lock:
            with self._lock:
                context.check_cancelled()
                snapshot = next_snapshot(self._snapshot, expected_revision, todos, change_reason)
                event = dict(snapshot, expected_revision=expected_revision, change_reason=change_reason)
                try:
                    self.store.record("todo_updated", invocation_id=context.invocation_id,
                                      tool_call_id=context.tool_call_id, tool=context.tool, **event)
                except PersistenceError:
                    context.cancel.set()
                    raise
                self._snapshot = snapshot
                result = json.loads(json.dumps(snapshot, ensure_ascii=False))
            context.emit("todo_updated", persist=False, **json.loads(json.dumps(event, ensure_ascii=False)))
        return result

    def summary(self):
        return summarize(self.read())


def todo_tools(workspace):
    from .tools import Tool

    def read(context):
        return context.store.todos.read()

    def write(context, **arguments):
        return context.store.todos.write(context, **arguments)

    return [
        Tool("todo_read", "Read this run's plan and revision. Use after a todo_conflict before retrying.",
             {}, read, [], read_only=True, contextual=True, workspace=workspace, permission_category="read"),
        Tool("todo_write", "Replace the full plan using expected_revision (initially 0). Preserve IDs. "
             "At most one in_progress. blocked/cancelled need reason; removing unfinished or reopening "
             "terminal items needs change_reason. Completed is a progress claim, not proof of verification.",
             WRITE_PROPERTIES, write, WRITE_REQUIRED, validator=validate_write,
             contextual=True, workspace=workspace, permission_category="session_state"),
    ]
