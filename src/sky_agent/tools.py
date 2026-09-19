from dataclasses import dataclass
from fnmatch import fnmatchcase
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator, ValidationError
import regex as re

from .execution import ExecutionContext, ToolError, run_process
from .persistence import RunStore
from .workspace import replace_file

MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_ARGUMENT_BYTES = 2 * MAX_FILE_BYTES
TEXT_LIMIT = 16000
SKIP_DIRS = {".sky-agent", ".git", ".venv", "venv", "__pycache__", "node_modules", "build", "dist"}


@dataclass
class Tool:
    name: str
    description: str
    properties: dict
    handler: Callable[..., Any]
    required: list[str] | None = None
    read_only: bool = False
    validator: Callable[..., None] | None = None
    contextual: bool = False
    workspace: Path | None = None

    def __post_init__(self):
        Draft202012Validator.check_schema(self.parameters)
        self._validator = Draft202012Validator(self.parameters)

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": self.properties,
                "required": list(self.properties) if self.required is None else self.required,
                "additionalProperties": False}

    @property
    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters,
        }}

    def prepare(self, arguments: str) -> dict:
        if not isinstance(arguments, str) or len(arguments.encode("utf-8")) > MAX_ARGUMENT_BYTES:
            raise ToolError("invalid_arguments", "Arguments must be a bounded JSON string")
        try:
            def invalid_constant(value):
                raise ValueError(f"Non-finite JSON number: {value}")
            kwargs = json.loads(arguments, parse_constant=invalid_constant)
            self._validator.validate(kwargs)
        except (ValueError, ValidationError, RecursionError) as exc:
            message = exc.message if isinstance(exc, ValidationError) else str(exc)
            raise ToolError("invalid_arguments", message[:2000]) from exc
        if self.validator:
            self.validator(**kwargs)
        return kwargs

    def invoke(self, kwargs: dict, context: ExecutionContext) -> dict:
        context.check_cancelled()
        if self.contextual:
            value = self.handler(context=context, **kwargs)
        else:
            value = self.handler(**kwargs)
        return {"ok": True, "result": value}

    def execute(self, arguments: str) -> str:
        """Convenience entry point with the same scheduler, journal, and policy defaults."""
        from .runtime import ToolRunner
        from .workspace import WorkspaceLease
        root = self.workspace or Path.cwd()
        with WorkspaceLease(root):
            context = ExecutionContext(RunStore(root))
            runner = ToolRunner([self], context)
            context.emit("session_started", directory=str(context.store.directory))
            try:
                messages = runner.run([{"id": "direct", "type": "function", "function": {
                    "name": self.name, "arguments": arguments,
                }}], round_number=1)
            except BaseException:
                context.emit("session_finished", status="failed")
                raise
            context.emit("session_finished", status="completed")
        return messages[0]["content"]


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def workspace_tools(workspace: Path, *, command_timeout: float = 30) -> list[Tool]:
    root = workspace.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Workspace must be a directory")
    if not 0 < command_timeout <= 3600:
        raise ValueError("command_timeout must be in (0, 3600]")

    def resolve(path: str) -> Path:
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            raise ToolError("path_denied", "Path must stay inside the workspace")
        relative = target.relative_to(root)
        if any(part.lower() == ".sky-agent" for part in relative.parts):
            raise ToolError("path_denied", "Runtime storage is reserved; use read_artifact")
        if os.name == "nt" and any(":" in part or part.endswith((".", " ")) for part in relative.parts):
            raise ToolError("path_denied", "Ambiguous Windows paths are not supported")
        return target

    def data_at(path: str) -> tuple[Path, bytes]:
        target = resolve(path)
        with target.open("rb") as file:
            data = file.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ToolError("file_too_large", f"File exceeds {MAX_FILE_BYTES} bytes")
        return target, data

    def check_path(path: str = ".", **kwargs):
        resolve(path)

    def check_read(path, **kwargs):
        if not resolve(path).is_file():
            raise ToolError("invalid_arguments", "path must identify an existing file")

    def check_write(path, content, overwrite=False, expected_hash=None):
        target = resolve(path)
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise ToolError("file_too_large", "New file is too large")
        if overwrite and expected_hash is None:
            raise ToolError("invalid_arguments", "overwrite requires expected_hash from read_file")
        if not overwrite and expected_hash is not None:
            raise ToolError("invalid_arguments", "expected_hash requires overwrite=true")
        if overwrite:
            check_version(path, expected_hash)
        elif target.exists():
            raise ToolError("file_exists", "Use overwrite=true with expected_hash to replace an existing file")

    def check_edit(path, old, new, expected_hash, replace_all=False):
        _, data = check_version(path, expected_hash)
        count = data.decode("utf-8").count(old)
        if count == 0 or (count != 1 and not replace_all):
            raise ToolError("edit_conflict", f"Expected one match, found {count}; use replace_all for multiple matches")

    def check_command(argv, cwd=".", **kwargs):
        if not resolve(cwd).is_dir():
            raise ToolError("invalid_arguments", "cwd must be a directory")
        if any("\0" in arg for arg in argv):
            raise ToolError("invalid_arguments", "Command arguments cannot contain NUL")

    def check_version(path: str, expected_hash: str) -> tuple[Path, bytes]:
        target, data = data_at(path)
        if file_hash(data) != expected_hash:
            raise ToolError("stale_file", "File changed; read it again before editing")
        return target, data

    def iter_files(path, pattern, recursive, context):
        directory = resolve(path)
        if not directory.is_dir():
            raise ToolError("invalid_arguments", "path must be a directory")
        def walk_error(error):
            raise error
        for current, dirs, files in os.walk(directory, followlinks=False, onerror=walk_error):
            context.check_cancelled()
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS
                             and not (Path(current) / d).is_symlink()
                             and not getattr(Path(current) / d, "is_junction", lambda: False)())
            for name in sorted(files):
                target = Path(current) / name
                if target.is_symlink() or name.startswith(".sky-agent-"):
                    continue
                relative = target.relative_to(root).as_posix()
                if fnmatchcase(target.relative_to(directory).as_posix(), pattern):
                    resolve(relative)
                    yield relative
            if not recursive:
                dirs.clear()

    def list_files(context, path=".", pattern="*", recursive=True, offset=0, limit=100):
        found = []
        more = False
        for index, relative in enumerate(iter_files(path, pattern, recursive, context)):
            if index < offset:
                continue
            if len(found) == limit:
                more = True
                break
            found.append(relative)
        return {"files": found, "next_offset": offset + len(found) if more else None}

    def search_text(query, context, path=".", pattern="*", regex=False, offset=0, limit=50):
        try:
            expression = re.compile(query) if regex else None
        except re.error as exc:
            raise ToolError("invalid_arguments", f"Invalid regex: {exc}") from exc
        matches = []
        count = 0
        skipped = 0
        for relative in iter_files(path, pattern, True, context):
            try:
                _, data = data_at(relative)
                if b"\0" in data:
                    skipped += 1
                    continue
                text = data.decode("utf-8")
            except (UnicodeError, OSError, ToolError):
                skipped += 1
                continue
            for number, line in enumerate(text.splitlines(), 1):
                context.check_cancelled()
                try:
                    matched = expression.search(line, timeout=0.05) if expression else query in line
                except TimeoutError as exc:
                    raise ToolError("search_timeout", "Regex exceeded its per-line time limit") from exc
                if matched:
                    if count >= offset:
                        if len(matches) == limit:
                            return {"matches": matches, "next_offset": offset + limit, "skipped_files": skipped}
                        matches.append({"path": relative, "line": number, "text": line[:2000],
                                        "truncated": len(line) > 2000})
                    count += 1
        return {"matches": matches, "next_offset": None, "skipped_files": skipped}

    def read_file(path, start_line=1, limit=200, column=0):
        _, data = data_at(path)
        lines = data.decode("utf-8").splitlines(keepends=True)
        if start_line > len(lines) + 1:
            raise ToolError("invalid_arguments", "start_line exceeds end of file")
        if column and (start_line > len(lines) or column >= len(lines[start_line - 1])):
            raise ToolError("invalid_arguments", "column exceeds line length")
        parts = []
        budget = TEXT_LIMIT
        next_position = None
        for index in range(start_line - 1, min(len(lines), start_line - 1 + limit)):
            segment = lines[index][column if index == start_line - 1 else 0:]
            taken = segment[:budget]
            parts.append(taken)
            budget -= len(taken)
            if len(taken) < len(segment):
                next_position = {"start_line": index + 1,
                                 "column": (column if index == start_line - 1 else 0) + len(taken)}
                break
            next_position = {"start_line": index + 2, "column": 0} if index + 1 < len(lines) else None
            if not budget:
                break
        return {"path": path, "text": "".join(parts), "hash": file_hash(data),
                "start_line": start_line, "total_lines": len(lines), "next": next_position}

    def write_file(path, content, overwrite=False, expected_hash=None):
        data = content.encode("utf-8")
        target = resolve(path)

        def recheck():
            if resolve(path) != target:
                raise ToolError("path_denied", "Resolved path changed")
            check_write(path, content, overwrite, expected_hash)

        recheck()
        replace_file(target, data, recheck, creating=not overwrite)
        return {"path": path, "hash": file_hash(data), "bytes": len(data)}

    def edit_file(path, old, new, expected_hash, replace_all=False):
        target, data = check_version(path, expected_hash)
        text = data.decode("utf-8")
        count = text.count(old)
        if count == 0 or (count != 1 and not replace_all):
            raise ToolError("edit_conflict", f"Expected one match, found {count}; use replace_all for multiple matches")
        replacement = text.replace(old, new).encode("utf-8")
        if len(replacement) > MAX_FILE_BYTES:
            raise ToolError("file_too_large", "Edited file is too large")

        def recheck():
            current, _ = check_version(path, expected_hash)
            if current != target:
                raise ToolError("path_denied", "Resolved path changed")

        replace_file(target, replacement, recheck, creating=False)
        return {"path": path, "hash": file_hash(replacement), "replacements": count}

    def run_command(argv, context, cwd=".", timeout=None):
        check_command(argv, cwd)
        return run_process(argv, resolve(cwd), timeout or command_timeout, context)

    def read_artifact(artifact_id, context, offset=0, limit=TEXT_LIMIT):
        return context.store.read_artifact(artifact_id, offset, limit)

    string = {"type": "string"}
    path_schema = {"type": "string", "minLength": 1, "maxLength": 4096}
    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    paging = {"offset": {"type": "integer", "minimum": 0},
              "limit": {"type": "integer", "minimum": 1, "maximum": 100}}
    discovery = {"path": path_schema, "pattern": {"type": "string", "maxLength": 4096}, **paging}
    boolean = {"type": "boolean"}

    def tool(name, description, properties, handler, required, *, read_only=False, contextual=False, validator=check_path):
        return Tool(name, description, properties, handler, required, read_only, validator, contextual, root)

    return [
        tool("list_files", "List workspace files in stable order; use next_offset for more. Generated directories are skipped.",
             {**discovery, "recursive": boolean}, list_files, [], read_only=True, contextual=True),
        tool("search_text", "Find literal text or regex in UTF-8 files. Returns paths and 1-based lines; follow next_offset.",
             {**discovery, "query": {"type": "string", "minLength": 1, "maxLength": 4096}, "regex": boolean},
             search_text, ["query"], read_only=True, contextual=True),
        tool("read_file", "Read UTF-8 text with its hash. Follow next.start_line/column for more; use hash when editing.",
             {"path": path_schema, "start_line": {"type": "integer", "minimum": 1},
              "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
              "column": {"type": "integer", "minimum": 0}}, read_file, ["path"], read_only=True, validator=check_read),
        tool("write_file", "Create a UTF-8 file. Replacing requires overwrite=true and the hash returned by read_file.",
             {"path": path_schema, "content": string, "overwrite": boolean, "expected_hash": digest},
             write_file, ["path", "content"], validator=check_write),
        tool("edit_file", "Replace an exact nonempty string after checking expected_hash. Multiple matches require replace_all=true.",
             {"path": path_schema, "old": {"type": "string", "minLength": 1}, "new": string,
              "expected_hash": digest, "replace_all": boolean}, edit_file, ["path", "old", "new", "expected_hash"], validator=check_edit),
        tool("read_artifact", "Read command output from an artifact_id returned by a tool in this session. Offsets count characters.",
             {"artifact_id": path_schema, "offset": {"type": "integer", "minimum": 0},
              "limit": {"type": "integer", "minimum": 1, "maximum": TEXT_LIMIT}},
             read_artifact, ["artifact_id"], read_only=True, contextual=True, validator=None),
        tool("run_command", "Run argv without an implicit shell, in workspace-relative cwd. Streams output to artifacts; failures include output details.",
             {"argv": {"type": "array", "minItems": 1, "maxItems": 256, "items": string},
              "cwd": path_schema, "timeout": {"type": "number", "exclusiveMinimum": 0, "maximum": 3600}},
             run_command, ["argv"], contextual=True, validator=check_command),
    ]
