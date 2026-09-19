from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import json
import time
from typing import Callable

from .execution import ExecutionContext, ToolError
from .persistence import PersistenceError
from .tools import Tool


class ProtocolError(ValueError):
    pass


@dataclass
class PermissionPolicy:
    read_only: bool = False
    denied: set[str] = field(default_factory=set)
    ask: set[str] = field(default_factory=set)
    approve: Callable[[str, dict], bool] | None = None

    def check(self, tool: Tool, arguments: dict):
        if tool.name in self.denied or (self.read_only and not tool.read_only):
            raise ToolError("permission_denied", f"Policy denies {tool.name}")
        if tool.name in self.ask and (self.approve is None or not self.approve(tool.name, arguments)):
            raise ToolError("permission_denied", f"Approval not granted for {tool.name}")


class ToolRunner:
    def __init__(self, tools: list[Tool], context: ExecutionContext, *, max_parallel: int = 4,
                 policy: PermissionPolicy | None = None):
        if not 1 <= max_parallel <= 32:
            raise ValueError("max_parallel must be between 1 and 32")
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("Tool names must be unique")
        self.context = context
        self.max_parallel = max_parallel
        self.policy = policy or PermissionPolicy()

    def validate_calls(self, calls: list[dict]):
        if not isinstance(calls, list) or len(calls) > 128:
            raise ProtocolError("tool_calls must be an array with at most 128 calls")
        ids = set()
        for call in calls:
            if not isinstance(call, dict):
                raise ProtocolError("Tool call must be an object")
            identifier = call.get("id")
            function = call.get("function")
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ProtocolError("Tool call IDs must be nonempty and unique within a round")
            if call.get("type") != "function" or not isinstance(function, dict):
                raise ProtocolError("Only function calls are supported")
            if not isinstance(function.get("name"), str) or not function["name"]:
                raise ProtocolError("Tool name must be a nonempty string")
            if not isinstance(function.get("arguments"), str):
                raise ProtocolError("Tool arguments must be a JSON string")
            ids.add(identifier)

    def batches(self, calls):
        pending = []
        for call in calls:
            tool = self.tools.get(call["function"]["name"])
            if tool and tool.read_only:
                pending.append(call)
            else:
                if pending:
                    yield pending
                    pending = []
                yield [call]
        if pending:
            yield pending

    def execute(self, call, round_number):
        name = call["function"]["name"]
        context = replace(self.context, invocation_id=f"{round_number}:{call['id']}",
                          tool_call_id=call["id"], tool=name)
        started = time.monotonic()
        context.emit("started", arguments=call["function"]["arguments"])
        try:
            context.check_cancelled()
            tool = self.tools.get(name)
            if tool is None:
                raise ToolError("unknown_tool", f"Unknown tool: {name}")
            kwargs = tool.prepare(call["function"]["arguments"])
            self.policy.check(tool, kwargs)
            result = tool.invoke(kwargs, context)
            json.dumps(result, allow_nan=False)
        except ToolError as exc:
            result = exc.result()
        except KeyboardInterrupt:
            self.context.cancel.set()
            result = ToolError("cancelled", "Execution interrupted").result()
        except PersistenceError:
            self.context.cancel.set()
            raise
        except Exception as exc:
            result = ToolError("tool_error", f"{type(exc).__name__}: {exc}"[:2000]).result()
        status = "completed" if result["ok"] else (
            "cancelled" if result["error"]["code"] == "cancelled" else "failed")
        context.emit("finished", result=result, status=status, duration=time.monotonic() - started)
        return {"role": "tool", "tool_call_id": call["id"],
                "content": json.dumps(result, ensure_ascii=False)}

    def run(self, calls, *, round_number: int):
        self.validate_calls(calls)
        messages = []
        pool = ThreadPoolExecutor(max_workers=self.max_parallel)
        try:
            for batch in self.batches(calls):
                if len(batch) == 1:
                    messages.append(self.execute(batch[0], round_number))
                else:
                    futures = [pool.submit(self.execute, call, round_number) for call in batch]
                    messages.extend(future.result() for future in futures)
            return messages
        except BaseException:
            self.context.cancel.set()
            raise
        finally:
            pool.shutdown(wait=True)
