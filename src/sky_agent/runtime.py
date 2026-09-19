from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import time

from .execution import ExecutionContext, ToolError
from .hooks import HookManager, ToolRequest, conversation_snapshot
from .permissions import PermissionPolicy
from .persistence import PersistenceError
from .tools import Tool


class ProtocolError(ValueError):
    pass


class ToolRunner:
    def __init__(self, tools: list[Tool], context: ExecutionContext, *, max_parallel: int = 4,
                 policy: PermissionPolicy | None = None, hooks: tuple[object, ...] = (),
                 lifecycle: HookManager | None = None):
        if not 1 <= max_parallel <= 32:
            raise ValueError("max_parallel must be between 1 and 32")
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("Tool names must be unique")
        self.context = context
        self.max_parallel = max_parallel
        self.policy = policy or PermissionPolicy()
        self.hooks = tuple(hooks)
        self.lifecycle = lifecycle or HookManager(self.hooks)

    @property
    def policy(self):
        return self._policy

    @policy.setter
    def policy(self, policy):
        self._policy = policy
        self._permission_hook = policy.new_session()

    def before_tool(self, request, context):
        decision = self.lifecycle.before_tool(request, context)
        self._permission_hook.before_tool(request, context, decision)

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

    def execute(self, call, round_number, conversation_json='{"messages":[],"truncated":false}'):
        name = call["function"]["name"]
        context = replace(self.context, invocation_id=f"{round_number}:{call['id']}",
                          tool_call_id=call["id"], tool=name)
        started = time.monotonic()
        request = None
        context.emit("started", arguments=call["function"]["arguments"])
        try:
            context.check_cancelled()
            tool = self.tools.get(name)
            if tool is None:
                raise ToolError("unknown_tool", f"Unknown tool: {name}")
            kwargs = tool.prepare(call["function"]["arguments"])
            request = ToolRequest.create(tool, kwargs, conversation_json)
            self.before_tool(request, context)
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
        if not result["ok"]:
            self.lifecycle.observers("tool_error", context,
                                     request=self.request_data(request), result=result, status=status)
        self.lifecycle.observers("after_tool", context, request=self.request_data(request),
                                 result=result, status=status)
        return {"role": "tool", "tool_call_id": call["id"],
                "content": json.dumps(result, ensure_ascii=False)}

    @staticmethod
    def request_data(request):
        if request is None:
            return None
        return {"tool_name": request.tool_name, "arguments": request.arguments,
                "fingerprint": request.fingerprint, "workspace": request.workspace}

    def run(self, calls, *, round_number: int, messages: list[dict] | None = None):
        self.validate_calls(calls)
        conversation_json = conversation_snapshot(messages or [])
        results = []
        pool = ThreadPoolExecutor(max_workers=self.max_parallel)
        try:
            for batch in self.batches(calls):
                if len(batch) == 1:
                    results.append(self.execute(batch[0], round_number, conversation_json))
                else:
                    futures = [pool.submit(self.execute, call, round_number, conversation_json) for call in batch]
                    results.extend(future.result() for future in futures)
            return results
        except BaseException:
            self.context.cancel.set()
            raise
        finally:
            pool.shutdown(wait=True)
