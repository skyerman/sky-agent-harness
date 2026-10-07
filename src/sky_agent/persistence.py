import json
import os
import re
from pathlib import Path
import threading
from datetime import datetime, timezone
from uuid import uuid4


class PersistenceError(RuntimeError):
    pass


class RunStore:
    def __init__(self, workspace: Path, *, parent=None, agent_id="root", parent_invocation_id=None):
        from .todos import TodoStore

        self.session_id = uuid4().hex
        self.directory = (parent.directory / "children" / self.session_id if parent else
                          workspace / ".sky-agent" / "runs" / self.session_id)
        self.agent_id = agent_id
        self.parent_session_id = parent.session_id if parent else None
        self.parent_invocation_id = parent_invocation_id
        self.artifacts = self.directory / "artifacts"
        self.artifacts.mkdir(parents=True)
        self._lock = threading.Lock()
        self.todos = TodoStore(self)

    def record(self, kind: str, **data) -> dict:
        event = {"kind": kind, "session_id": self.session_id,
                 "timestamp": datetime.now(timezone.utc).isoformat(), "agent_id": self.agent_id,
                 "parent_session_id": self.parent_session_id,
                 "parent_invocation_id": self.parent_invocation_id, **data}
        try:
            with self._lock:
                with (self.directory / "events.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(event, ensure_ascii=False) + "\n")
                    file.flush()
                    os.fsync(file.fileno())
        except (OSError, TypeError, ValueError) as exc:
            raise PersistenceError(f"Could not persist {kind}: {exc}") from exc
        return event

    def new_artifact(self, suffix: str) -> tuple[str, Path]:
        name = f"{uuid4().hex}.{suffix}.txt"
        return name, self.artifacts / name

    def read_artifact(self, artifact_id: str, offset: int = 0, limit: int = 16000) -> dict:
        if not re.fullmatch(r"[0-9a-f]{32}\.[a-z]+\.txt", artifact_id):
            raise ValueError("Invalid artifact ID")
        path = (self.artifacts / artifact_id).resolve()
        if path.parent != self.artifacts.resolve() or not path.is_file():
            raise ValueError("Artifact does not exist in this session")
        with path.open(encoding="utf-8", newline="") as file:
            remaining = offset
            while remaining:
                skipped = file.read(min(remaining, 65536))
                if not skipped:
                    break
                remaining -= len(skipped)
            text = file.read(limit)
            more = bool(file.read(1))
        return {"artifact_id": artifact_id, "text": text,
                "next_offset": offset + len(text) if more else None}


def inspect_run(directory: Path) -> dict:
    from .execution import ToolError
    from .todos import replay_update, summarize

    events = []
    lines = (directory / "events.jsonl").read_bytes().splitlines(keepends=True)
    incomplete_tail = False
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
            if not isinstance(event, dict) or "kind" not in event:
                raise ValueError("Invalid journal event")
            events.append(event)
        except (ValueError, UnicodeDecodeError):
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                incomplete_tail = True
                break
            raise ValueError(f"Corrupt journal at line {index + 1}") from None
    calls = {}
    messages = []
    hooks = []
    cleanup_errors = []
    todos = {"revision": 0, "todos": []}
    todo_history = []
    subagents = {}
    skills = {"catalogue": None, "loaded": {}, "resources": []}
    identity = {}
    model_calls = 0
    status = "unknown"
    for index, event in enumerate(events):
        try:
            if not identity:
                identity = {key: event.get(key) for key in ("session_id", "agent_id", "parent_session_id", "parent_invocation_id")}
            if event["kind"] == "started":
                if event["invocation_id"] in calls:
                    raise ValueError("Duplicate invocation")
                calls[event["invocation_id"]] = {
                    "tool": event["tool"], "tool_call_id": event["tool_call_id"], "status": "unknown",
                }
            elif event["kind"] == "artifacts":
                calls[event["invocation_id"]]["artifacts"] = event["artifacts"]
            elif event["kind"] == "finished":
                calls[event["invocation_id"]].update(status=event["status"], result=event["result"])
            elif event["kind"] in {"permission_requested", "permission_classification", "permission_decision"}:
                calls[event["invocation_id"]].setdefault("permissions", []).append(event)
            elif event["kind"] == "message":
                messages.append(event["message"])
            elif event["kind"] == "session_finished":
                status = event["status"]
                cleanup_errors = event.get("cleanup_errors", [])
            elif event["kind"] in {"hook_started", "hook_finished", "hook_failed"}:
                hooks.append(event)
            elif event["kind"] == "todo_updated":
                todos = replay_update(todos, event)
                todo_history.append(event)
            elif event["kind"] == "subagent_queued":
                if event["agent_id"] in subagents:
                    raise ValueError("Duplicate child identity")
                subagents[event["agent_id"]] = {"name": event["name"], "profile": event["profile"],
                                              "status": "queued", "parent_invocation_id": event["invocation_id"]}
            elif event["kind"] in {"subagent_started", "subagent_finished"}:
                subagents[event["agent_id"]].update(event)
                if event["kind"] == "subagent_started":
                    subagents[event["agent_id"]]["status"] = "running"
            elif event["kind"] == "model_call_reserved" and event.get("purpose") == "model":
                model_calls += 1
            elif event["kind"] == "skill_catalogue":
                skills["catalogue"] = {key: event[key] for key in ("skills", "total", "next_offset")}
            elif event["kind"] == "skill_loaded":
                if event["skill_id"] in skills["loaded"]:
                    raise ValueError("Duplicate skill activation")
                skills["loaded"][event["skill_id"]] = event
            elif event["kind"] == "skill_resource_read":
                if event["skill_id"] not in skills["loaded"]:
                    raise ValueError("Resource read before skill activation")
                skills["resources"].append(event)
        except (KeyError, TypeError, ValueError, ToolError) as exc:
            raise ValueError(f"Invalid journal record at line {index + 1}: {exc}") from exc
    return {"directory": str(directory.resolve()), "status": status,
            "incomplete_tail": incomplete_tail, "calls": calls, "messages": messages,
            "hooks": hooks, "cleanup_errors": cleanup_errors,
            "todos": todos, "todo_history": todo_history, "todo_summary": summarize(todos),
            "identity": identity, "subagents": subagents, "model_calls": model_calls, "skills": skills}
