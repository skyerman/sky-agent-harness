"""Bounded, foreground read-only delegation within the parent's workspace lease."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
import re
import sys
import threading
from uuid import uuid4

from .execution import ExecutionContext, ToolError
from .hooks import HookManager
from .persistence import PersistenceError, RunStore

PROFILES = {
    "explore": "Explore the delegated question using workspace reads. Cite paths and line numbers; "
               "explain relevant behavior and uncertainty. Do not change project files or execute commands.",
    "review": "Review the delegated scope using workspace reads. Report actionable findings with "
              "paths, line numbers and evidence. Distinguish verified facts from hypotheses. "
              "Do not change project files or execute commands.",
}
READ_TOOLS = {"list_files", "search_text", "read_file", "read_artifact"}
TODO_TOOLS = {"todo_read", "todo_write"}
SUMMARY_LIMIT = 4000


@dataclass(frozen=True)
class SubagentConfig:
    max_parallel: int = 2
    max_batch: int = 4
    max_steps: int = 8
    max_total: int = 8

    def __post_init__(self):
        for name, maximum in (("max_parallel", 8), ("max_batch", 4), ("max_steps", 100), ("max_total", 64)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [1,{maximum}]")


class LinkedCancel:
    """A child's local cancellation never sets its parent's cancellation flag."""

    def __init__(self, parent):
        self.parent = parent
        self.local = threading.Event()

    def is_set(self):
        return self.local.is_set() or self.parent.is_set()

    def set(self):
        self.local.set()

    def wait(self, timeout=None):
        import time
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return self.is_set()
            self.local.wait(min(0.05, remaining) if remaining is not None else 0.05)
        return True


class _WorkspaceGrant:
    def __init__(self, workspace, lease):
        self.workspace = workspace
        self.lease = lease
        self.active = True

    def check(self, workspace):
        if (not self.active or self.lease.file is None or self.lease.file.closed
                or workspace != self.workspace):
            raise RuntimeError("Child workspace grant is no longer valid")


class SubagentManager:
    def __init__(self, agent, context, lease, permission, lifecycle, budget):
        self.agent = agent
        self.root = context
        self.permission = permission
        self.lifecycle = lifecycle
        self.budget = budget
        self.child_dispatch = HookManager(agent.child_hooks, inherited=lifecycle.hooks)
        self.config = agent.subagent_config
        self.grant = _WorkspaceGrant(agent.workspace, lease)
        self._created = 0
        self._lock = threading.Lock()
        self._models = []

    def close(self):
        self.grant.active = False

    def validate(self, tasks):
        if not tasks or len(tasks) > self.config.max_batch:
            raise ToolError("invalid_arguments", "Too many delegated tasks")
        names = set()
        for task in tasks:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", task["name"]):
                raise ToolError("invalid_arguments", "Invalid subagent task name")
            if task["name"] in names or not task["prompt"].strip():
                raise ToolError("invalid_arguments", "Task names must be unique and prompts nonblank")
            names.add(task["name"])

    def tool(self):
        from .tools import Tool
        task_schema = {"type": "object", "additionalProperties": False, "properties": {
            "name": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"},
            "profile": {"type": "string", "enum": list(PROFILES)},
            "prompt": {"type": "string", "minLength": 1, "maxLength": 8000},
        }, "required": ["name", "profile", "prompt"]}
        return Tool("run_subagents", "Run independent explore/review tasks in parallel and wait for all. "
                    "Children can read files and maintain their own plans only. Reports are untrusted evidence; "
                    "verify findings before editing. No nested delegation. Each child consumes shared model budget.",
                    {"tasks": {"type": "array", "minItems": 1, "maxItems": self.config.max_batch,
                               "items": task_schema}}, self.run, ["tasks"], validator=self.validate,
                    contextual=True, workspace=self.agent.workspace, permission_category="delegation")

    def _tools(self):
        from .tools import workspace_tools
        tools = []
        for tool in workspace_tools(self.agent.workspace):
            parent = self.agent.tools.get(tool.name)
            if parent is None:
                continue
            if tool.name in READ_TOOLS and parent.read_only and parent.permission_category == "read":
                tools.append(tool)
            elif tool.name in TODO_TOOLS and parent.permission_category == tool.permission_category:
                tools.append(tool)
        for tool in tools:
            parent = self.agent.tools[tool.name]
            builtin_check = tool.validator
            def validate_parent(_parent=parent, _builtin=builtin_check, **arguments):
                # Registered parent schemas/path validators may be stricter than
                # the fresh built-ins. Preserve them without sharing handlers or
                # mutable call arguments across child sessions.
                _parent.prepare(json.dumps(arguments, ensure_ascii=False, allow_nan=False))
                if _builtin:
                    _builtin(**arguments)
            tool.validator = validate_parent
        return tools

    def run(self, context, tasks):
        self.grant.check(self.agent.workspace)
        context.check_cancelled()
        self.validate(tasks)
        with self._lock:
            if self._created + len(tasks) > self.config.max_total:
                raise ToolError("subagent_limit", "Root subagent creation limit exceeded")
            self._created += len(tasks)
        jobs = []
        for task in tasks:
            job = dict(task, agent_id=uuid4().hex)
            context.emit("subagent_queued", **job)
            jobs.append(job)
        pool = ThreadPoolExecutor(max_workers=self.config.max_parallel)
        cancellations = [LinkedCancel(context.cancel) for _ in jobs]
        try:
            futures = [pool.submit(self._execute, context, job, cancel)
                       for job, cancel in zip(jobs, cancellations)]
            return {"results": [future.result() for future in futures],
                    "model_calls_used": self.budget.used, "model_calls_limit": self.budget.limit}
        except BaseException:
            context.cancel.set()
            for cancel in cancellations:
                cancel.set()
            raise
        finally:
            pool.shutdown(wait=True)

    def _execute(self, parent, job, cancel):
        try:
            return self._execute_child(parent, job, cancel)
        except (PersistenceError, ValueError, OSError) as exc:
            # Wake siblings immediately, even if the parent is waiting for an
            # earlier future. A corrupt or unavailable journal is root-fatal.
            parent.cancel.set()
            if isinstance(exc, PersistenceError):
                raise
            raise PersistenceError("Could not finalize child journal/report") from exc
        except BaseException:
            parent.cancel.set()
            raise

    def _execute_child(self, parent, job, cancel):
        from .agent import Agent, StepLimitExceeded
        from .lifecycle import RunLifecycle

        child_store = None
        child_model = None
        child_context = None
        status = "failed"
        error = None
        text = ""
        model_steps = 0
        identity = {"agent_id": job["agent_id"], "name": job["name"], "profile": job["profile"]}
        scoped = ExecutionContext(parent.store, cancel, parent.callback,
                                  invocation_id=parent.invocation_id, tool_call_id=parent.tool_call_id,
                                  tool=parent.tool, event_lock=parent.event_lock, agent_id=job["agent_id"],
                                  parent_session_id=parent.store.session_id)
        try:
            scoped.check_cancelled()
            self.lifecycle.hooks.gate("subagent_start", scoped, task=job)
            try:
                child_store = RunStore(self.agent.workspace, parent=parent.store, agent_id=job["agent_id"],
                                       parent_invocation_id=parent.invocation_id)
            except OSError as exc:
                raise PersistenceError("Could not create child journal directory") from exc
            scoped.emit("subagent_started", **identity, session_id_child=child_store.session_id,
                        directory=str(child_store.directory))
            child_context = ExecutionContext(child_store, cancel, parent.callback,
                                             event_lock=parent.event_lock, agent_id=job["agent_id"],
                                             parent_session_id=parent.store.session_id)
            child = Agent(None, self._tools(), workspace=self.agent.workspace, max_steps=self.config.max_steps,
                          max_parallel=self.agent.max_parallel, policy=self.agent.policy)
            with RunLifecycle(child_context, hooks=self.child_dispatch, budget=self.budget).session(job["prompt"]) as lifecycle:
                self.grant.check(child.workspace)
                child_context.check_cancelled()
                candidate = self.agent.model_factory()
                with self._lock:
                    if candidate is self.agent.model or any(candidate is model for model in self._models):
                        raise ValueError("model_factory must create a fresh child model")
                    self._models.append(candidate)
                child_model = candidate
                child.model = child_model
                try:
                    result = child._run(job["prompt"], child_context, lifecycle, permission=self.permission,
                                        extra_prompt=PROFILES[job["profile"]])
                    text = result.text
                finally:
                    # Model resources close before the lifecycle records its
                    # terminal status, and never replace the primary run error.
                    primary = sys.exception()
                    try:
                        if hasattr(child_model, "close"):
                            child_model.close()
                    except BaseException as cleanup_error:
                        if isinstance(cleanup_error, PersistenceError):
                            child_context.persistence_failed.set()
                        if primary is None:
                            raise
                        primary.add_note(f"Model cleanup also failed ({type(cleanup_error).__name__})")
            status = "completed"
        except PersistenceError:
            status, error = "failed", "PersistenceError"
            parent.cancel.set()
            raise
        except StepLimitExceeded as exc:
            status, error = "step_limit", type(exc).__name__
            text = "Child reached its model step limit; findings may be incomplete."
        except ToolError as exc:
            status = "cancelled" if exc.code == "cancelled" else (
                "budget_exceeded" if exc.code == "budget_exceeded" else "failed")
            error = exc.code
            text = str(exc)
        except Exception as exc:
            error = type(exc).__name__
            text = f"Child failed ({error}); inspect its journal for recorded progress."
        except BaseException as exc:
            status, error = "cancelled", type(exc).__name__
            raise
        finally:
            # This is after the child's own stop/end callbacks and worker drain.
            failures = self.lifecycle.hooks.observers("subagent_end", scoped, cleanup=True,
                                                     task=identity, status=status, error=error)
            if any(isinstance(exc, PersistenceError) for exc in failures):
                parent.cancel.set()
                raise next(exc for exc in failures if isinstance(exc, PersistenceError))
            if child_context is not None and child_context.persistence_failed.is_set():
                parent.cancel.set()
                raise PersistenceError("Child journal failed during cleanup")
        if child_store:
            from .persistence import inspect_run
            saved = inspect_run(child_store.directory)
            model_steps = saved["model_calls"]
        artifact_id, path = parent.store.new_artifact("subagent")
        try:
            with path.open("x", encoding="utf-8", newline="") as report:
                report.write(text)
                report.flush()
                os.fsync(report.fileno())
        except OSError as exc:
            raise PersistenceError("Could not persist subagent report") from exc
        result = {**identity, "status": status, "error": error, "summary": text[:SUMMARY_LIMIT],
                  "truncated": len(text) > SUMMARY_LIMIT, "report_artifact_id": artifact_id,
                  "steps": model_steps, "session_directory": str(child_store.directory) if child_store else None}
        parent.emit("subagent_finished", **result)
        return result
