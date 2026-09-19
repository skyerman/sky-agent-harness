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
    def __init__(self, workspace: Path):
        self.session_id = uuid4().hex
        self.directory = workspace / ".sky-agent" / "runs" / self.session_id
        self.artifacts = self.directory / "artifacts"
        self.artifacts.mkdir(parents=True)
        self._lock = threading.Lock()

    def record(self, kind: str, **data) -> dict:
        event = {"kind": kind, "session_id": self.session_id,
                 "timestamp": datetime.now(timezone.utc).isoformat(), **data}
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
    status = "unknown"
    for index, event in enumerate(events):
        try:
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
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid journal record at line {index + 1}: {exc}") from exc
    return {"directory": str(directory.resolve()), "status": status,
            "incomplete_tail": incomplete_tail, "calls": calls, "messages": messages}
