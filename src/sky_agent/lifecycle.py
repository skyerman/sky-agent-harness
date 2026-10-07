from contextlib import contextmanager

from .execution import ToolError
from .hooks import HookManager
from .persistence import PersistenceError


class RunLifecycle:
    """Own model boundaries and one-way session finalization outside the loop."""

    def __init__(self, context, hooks=(), *, budget=None):
        self.context = context
        self.hooks = hooks if isinstance(hooks, HookManager) else HookManager(hooks)
        self.budget = budget

    @contextmanager
    def session(self, task):
        context = self.context
        primary = None
        status = "completed"
        try:
            context.emit("session_started", directory=str(context.store.directory))
            context.check_cancelled()
            self.hooks.gate("session_start", context, task=task)
            self.hooks.gate("user_prompt", context, task=task)
            context.check_cancelled()
            yield self
        except BaseException as exc:
            primary = exc
            context.cancel.set()
            status = "cancelled" if isinstance(exc, KeyboardInterrupt) or (
                isinstance(exc, ToolError) and exc.code == "cancelled") else "failed"
            failures = self.hooks.observers("error", context, cleanup=True,
                                            error=type(exc).__name__, status=status)
            if any(isinstance(failure, PersistenceError) for failure in failures):
                context.persistence_failed.set()
            raise
        finally:
            # ToolRunner drains its pool and command process trees before control
            # reaches here. Cleanup ignores cancellation and attempts every hook.
            failures = self.hooks.stop(context, reason=status,
                                       error=type(primary).__name__ if primary else None)
            if primary is None and failures:
                status = "failed"
            error = primary if primary is not None else (failures[0] if failures else None)
            failures.extend(self.hooks.observers(
                "session_end", context, cleanup=True, status=status,
                error=type(error).__name__ if error is not None else None))
            if primary is None and failures:
                status = "failed"
            error = primary if primary is not None else (failures[0] if failures else None)
            try:
                context.emit("session_finished", status=status,
                             error=type(error).__name__ if error is not None else None,
                             cleanup_errors=[type(exc).__name__ for exc in failures],
                             todo_summary=context.store.todos.summary(),
                             model_calls_used=self.budget.used if self.budget else None)
            except BaseException as exc:
                failures.append(exc)
            if failures:
                if any(isinstance(failure, PersistenceError) for failure in failures):
                    context.persistence_failed.set()
                if primary is not None:
                    primary.add_note("Cleanup also failed: " + ", ".join(type(exc).__name__ for exc in failures))
                else:
                    raise failures[0]

    def complete(self, model, messages, schemas, step):
        context = self.context
        context.check_cancelled()
        self.hooks.gate("before_model", context, step=step, messages=messages, tools=schemas)
        context.check_cancelled()
        try:
            if self.budget:
                self.budget.reserve(context)
            response = model.complete(messages, schemas)
        except BaseException as exc:
            failures = self.hooks.observers("model_error", context, cleanup=True,
                                            step=step, error=type(exc).__name__)
            if any(isinstance(failure, PersistenceError) for failure in failures):
                context.persistence_failed.set()
            raise
        context.store.record("message", message=response)
        self.hooks.observers("after_model", context, step=step, response=response)
        context.check_cancelled()
        return response
