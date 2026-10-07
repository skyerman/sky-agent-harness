import threading

from .execution import ToolError


class ModelBudget:
    """Atomic upper bound on attempted provider requests across a root run."""

    def __init__(self, limit):
        if type(limit) is not int or not 1 <= limit <= 10000:
            raise ValueError("model call budget must be an integer in [1,10000]")
        self.limit = limit
        self._used = 0
        self._lock = threading.Lock()

    def reserve(self, context, purpose="model"):
        context.check_cancelled()
        with self._lock:
            if self._used >= self.limit:
                raise ToolError("budget_exceeded", "Root model-call budget exhausted")
            self._used += 1
            used = self._used
        context.emit("model_call_reserved", purpose=purpose, used=used, limit=self.limit)

    @property
    def used(self):
        with self._lock:
            return self._used
