import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from sky_agent import Agent, HookDecision, StepLimitExceeded
from sky_agent.execution import ExecutionContext, ToolError
from sky_agent.permissions import Classification, PermissionPolicy
from sky_agent.persistence import PersistenceError, RunStore, inspect_run
from sky_agent.runtime import ProtocolError, ToolRunner
from sky_agent.tools import Tool, workspace_tools
from sky_agent.workspace import WorkspaceLease


def call(name, args=None):
    return {"id": "call_1", "type": "function", "function": {
        "name": name, "arguments": json.dumps(args or {})}}


class Model:
    def complete(self, messages, tools):
        if len(messages) == 2 and tools:
            return {"role": "assistant", "content": None, "tool_calls": [call("action")]}
        return {"role": "assistant", "content": "done", "reasoning_content": "private"}


class DecisionHook:
    def __init__(self, decision):
        self.decision = decision

    def before_tool(self, request, context):
        return HookDecision(self.decision, "Custom " + self.decision)


class HookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.effects = []
        self.tool = Tool("action", "", {}, lambda: self.effects.append("ran"))

    def tearDown(self):
        self.temp.cleanup()

    def test_hook_allow_cannot_bypass_policy_ask(self):
        class Allow:
            def before_tool(self, request, context):
                return HookDecision("allow", "advisory")

        class Model:
            def complete(self, messages, tools):
                if len(messages) == 2:
                    return {"role": "assistant", "content": None, "tool_calls": [call("action")]}
                return {"role": "assistant", "content": "done"}

        agent = Agent(Model(), [self.tool], workspace=self.root,
                      policy=PermissionPolicy(mode="default"), hooks=(Allow(),))
        result = agent.run("run it")
        tool_result = json.loads(result.messages[3]["content"])
        self.assertEqual(tool_result["error"]["code"], "permission_denied")
        self.assertEqual(self.effects, [])

    def test_hook_ask_uses_human_approval_then_allows(self):
        class Ask:
            def before_tool(self, request, context):
                return HookDecision("ask", "custom gate")

        class Model:
            def complete(self, messages, tools):
                if len(messages) == 2:
                    return {"role": "assistant", "content": None, "tool_calls": [call("action")]}
                return {"role": "assistant", "content": "done"}

        agent = Agent(Model(), [self.tool], workspace=self.root,
                      policy=PermissionPolicy(mode="allow", human_approve=lambda request: True),
                      hooks=(Ask(),))
        self.assertEqual(agent.run("run it").text, "done")
        self.assertEqual(self.effects, ["ran"])

    def test_lifecycle_order_and_graceful_stop(self):
        events = []

        class Lifecycle:
            def session_start(self, context, **kwargs): events.append("session_start")
            def user_prompt(self, context, **kwargs): events.append("user_prompt")
            def before_model(self, context, **kwargs): events.append("before_model")
            def after_model(self, context, **kwargs): events.append("after_model")
            def after_tool(self, context, **kwargs): events.append("after_tool")
            def stop(self, context, **kwargs): events.append("stop")
            def session_end(self, context, **kwargs): events.append("session_end")

        class Model:
            def complete(self, messages, tools):
                if len(messages) == 2:
                    return {"role": "assistant", "content": None, "tool_calls": [call("action")]}
                return {"role": "assistant", "content": "done"}

        result = Agent(Model(), [self.tool], workspace=self.root, hooks=(Lifecycle(),)).run("run it")
        self.assertEqual(result.text, "done")
        self.assertLess(events.index("stop"), events.index("session_end"))
        self.assertEqual(events.count("stop"), 1)
        self.assertEqual(events[:4], ["session_start", "user_prompt", "before_model", "after_model"])
        self.assertEqual(self.effects, ["ran"])
        self.assertEqual(events, ["session_start", "user_prompt", "before_model", "after_model",
                                  "after_tool", "before_model", "after_model", "stop", "session_end"])

    def test_stop_failures_do_not_skip_later_stop_hooks(self):
        events = []

        class Broken:
            def stop(self, context, **kwargs):
                events.append("broken")
                raise RuntimeError("cleanup")

        class Later:
            def stop(self, context, **kwargs): events.append("later")

        class Model:
            def complete(self, messages, tools):
                return {"role": "assistant", "content": "done"}

        agent = Agent(Model(), [], workspace=self.root, hooks=(Broken(), Later()))
        with self.assertRaises(RuntimeError):
            agent.run("run it")
        self.assertEqual(events, ["broken", "later"])
        summary = inspect_run(agent.last_session_directory)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["cleanup_errors"], ["RuntimeError"])

    def test_hook_allow_preserves_every_policy_gate(self):
        policies = [PermissionPolicy(denied={"action"}, allowed={"action"}),
                    PermissionPolicy(ask={"action"}, allowed={"action"}),
                    PermissionPolicy(read_only=True), PermissionPolicy(mode="readOnly"),
                    PermissionPolicy(mode="acceptEdits"),
                    PermissionPolicy(mode="auto", classifier=Mock(classify=Mock(
                        return_value=Classification("deny", "No")))),
                    PermissionPolicy(mode="auto", classifier=Mock(classify=Mock(
                        return_value=Classification("ask", "Confirm"))))]
        for policy in policies:
            with self.subTest(mode=policy.mode, denied=policy.denied, ask=policy.ask):
                result = Agent(Model(), [self.tool], workspace=self.root, policy=policy,
                               hooks=(DecisionHook("allow"),)).run("run it")
                self.assertFalse(json.loads(result.messages[3]["content"])["ok"])
        self.assertEqual(self.effects, [])

    def test_hook_decisions_merge_before_any_prompt(self):
        for decisions in [("ask", "deny", "allow"), ("allow", "deny", "ask"),
                          ("allow", "ask"), ("ask", "allow")]:
            approve = Mock(return_value=False)
            result = Agent(Model(), [self.tool], workspace=self.root,
                           policy=PermissionPolicy(ask={"action"}, human_approve=approve),
                           hooks=tuple(DecisionHook(d) for d in decisions)).run("run it")
            self.assertFalse(json.loads(result.messages[3]["content"])["ok"])
            self.assertEqual(approve.call_count, 0 if "deny" in decisions else 1)
        self.assertEqual(self.effects, [])

    def test_multiple_asks_and_policy_ask_prompt_only_once(self):
        approve = Mock(return_value=True)
        result = Agent(Model(), [self.tool], workspace=self.root,
                       policy=PermissionPolicy(ask={"action"}, human_approve=approve),
                       hooks=(DecisionHook("ask"), DecisionHook("allow"), DecisionHook("ask"))).run("task")
        self.assertEqual(approve.call_count, 1)
        self.assertEqual(self.effects, ["ran"])
        self.assertTrue(json.loads(result.messages[3]["content"])["ok"])

    def test_snapshots_and_observer_errors_cannot_rewrite_results(self):
        seen = []

        class Mutate:
            def before_model(self, context, *, messages, tools, step):
                messages.clear()
                tools.clear()

            def after_model(self, context, *, response, step):
                seen.append("reasoning_content" in response)
                response.clear()

            def after_tool(self, context, *, result, **kwargs):
                result["ok"] = False
                raise RuntimeError("observer failed")

        agent = Agent(Model(), [self.tool], workspace=self.root, hooks=(Mutate(),))
        result = agent.run("task")
        self.assertEqual(result.text, "done")
        self.assertEqual(result.messages[-1]["reasoning_content"], "private")
        self.assertEqual(seen, [False, False])
        self.assertEqual(self.effects, ["ran"])
        self.assertTrue(json.loads(result.messages[3]["content"])["ok"])
        self.assertTrue(any(e["kind"] == "hook_failed" for e in inspect_run(agent.last_session_directory)["hooks"]))

    def test_gates_veto_without_starting_model_or_tool(self):
        for method in ("session_start", "user_prompt", "before_model"):
            with self.subTest(method=method):
                seen = []
                hook = type("Gate", (), {method: lambda *args, **kwargs: False,
                                         "stop": lambda *args, **kwargs: seen.append("stop"),
                                         "session_end": lambda *args, **kwargs: seen.append("end")})()
                model = Mock()
                agent = Agent(model, [], workspace=self.root, hooks=(hook,))
                with self.assertRaises(ToolError) as caught:
                    agent.run("task")
                self.assertEqual(caught.exception.code, "hook_failed")
                model.complete.assert_not_called()
                self.assertEqual(seen, ["stop", "end"])
                self.assertEqual(inspect_run(agent.last_session_directory)["status"], "failed")

    def test_stop_request_before_model_prevents_network(self):
        class Stop:
            def before_model(self, context, **kwargs):
                agent.request_stop()
        model = Mock()
        agent = Agent(model, [], workspace=self.root, hooks=(Stop(),))
        self.assertFalse(agent.request_stop())
        with self.assertRaises(ToolError) as caught:
            agent.run("task")
        self.assertEqual(caught.exception.code, "cancelled")
        model.complete.assert_not_called()
        self.assertFalse(agent.request_stop())

    def test_primary_exception_survives_all_cleanup_errors(self):
        seen = []

        class Broken:
            def model_error(self, context, **kwargs): seen.append("model_error")
            def error(self, context, **kwargs): seen.append("error")
            def stop(self, context, **kwargs):
                seen.append("stop")
                raise KeyboardInterrupt()
            def session_end(self, context, **kwargs):
                seen.append("end")
                raise RuntimeError("cleanup")

        original = ValueError("model failed")
        agent = Agent(Mock(complete=Mock(side_effect=original)), [], workspace=self.root, hooks=(Broken(),))
        with self.assertRaises(ValueError) as caught:
            agent.run("task")
        self.assertIs(caught.exception, original)
        self.assertEqual(seen, ["model_error", "error", "stop", "end"])
        self.assertEqual(inspect_run(agent.last_session_directory)["status"], "failed")
        with WorkspaceLease(self.root):
            pass

    def test_cleanup_runs_even_when_initial_journal_write_fails(self):
        seen = []
        class Cleanup:
            def stop(self, context, **kwargs): seen.append("stop")
            def session_end(self, context, **kwargs): seen.append("end")
        original = PersistenceError("disk full")
        agent = Agent(Mock(), [], workspace=self.root, hooks=(Cleanup(),))
        with patch.object(RunStore, "record", side_effect=original), self.assertRaises(PersistenceError) as caught:
            agent.run("task")
        self.assertIs(caught.exception, original)
        self.assertEqual(seen, ["stop", "end"])
        self.assertFalse(agent.request_stop())
        with WorkspaceLease(self.root):
            pass

    def test_step_limit_and_precancelled_sessions_stop_once(self):
        for cancelled in (False, True):
            seen = []
            class Cleanup:
                def stop(self, context, *, reason, error): seen.append(reason)
            cancel = threading.Event()
            if cancelled:
                cancel.set()
            agent = Agent(Model(), [self.tool], workspace=self.root, max_steps=1, hooks=(Cleanup(),))
            with self.assertRaises(ToolError if cancelled else StepLimitExceeded):
                agent.run("task", cancel=cancel)
            self.assertEqual(seen, ["cancelled" if cancelled else "failed"])

    def test_tool_errors_notify_after_persistence_without_replay(self):
        seen = []
        class Observe:
            def tool_error(self, context, *, result, **kwargs):
                saved = inspect_run(context.store.directory)["calls"][context.invocation_id]
                seen.append(("error", saved["result"] == result))
            def after_tool(self, context, **kwargs): seen.append(("after", True))
        self.tool.handler = Mock(side_effect=ToolError("custom", "failed"))
        Agent(Model(), [self.tool], workspace=self.root, hooks=(Observe(),)).run("task")
        self.assertEqual(seen, [("error", True), ("after", True)])
        self.assertEqual(self.tool.handler.call_count, 1)

    def test_hook_approvals_remain_serialized_during_parallel_reads(self):
        entered = threading.Event()
        active = 0
        peak = 0
        context = ExecutionContext(RunStore(self.root))
        def approve(request):
            nonlocal active, peak
            active += 1
            peak = max(active, peak)
            entered.set()
            request.cancel.wait(5)
            active -= 1
            return True
        self.tool.read_only = True
        runner = ToolRunner([self.tool], context, hooks=(DecisionHook("ask"),),
                            policy=PermissionPolicy(human_approve=approve))
        calls = [dict(call("action"), id=str(i)) for i in range(3)]
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(runner.run, calls, round_number=1)
            try:
                self.assertTrue(entered.wait(5))
            finally:
                context.cancel.set()
            results = future.result(5)
        self.assertEqual(peak, 1)
        self.assertEqual(self.effects, [])
        self.assertEqual([json.loads(r["content"])["error"]["code"] for r in results], ["cancelled"] * 3)

    def test_command_is_drained_and_results_saved_before_stop(self):
        seen = []
        tools = workspace_tools(self.root)
        class CommandModel:
            def complete(self, messages, schemas):
                return {"role": "assistant", "content": None, "tool_calls": [call("run_command", {
                    "argv": [sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(20)"]})]}
        class Cleanup:
            def stop(self, context, **kwargs):
                summary = inspect_run(context.store.directory)
                saved = summary["calls"]["1:call_1"]
                seen.append(saved["status"])
                seen.append(summary["messages"][-1]["role"])
                seen.append(context.store.read_artifact(saved["artifacts"]["stdout"])["text"])
        def observe(event):
            if event["kind"] == "output" and "ready" in event["text"]:
                agent.request_stop()
        agent = Agent(CommandModel(), tools, workspace=self.root, hooks=(Cleanup(),), on_event=observe)
        with self.assertRaises(ToolError) as caught:
            agent.run("task")
        self.assertEqual(caught.exception.code, "cancelled")
        self.assertEqual(seen[:2], ["cancelled", "tool"])
        self.assertIn("ready", seen[2])
        self.assertEqual(inspect_run(agent.last_session_directory)["status"], "cancelled")
        self.assertFalse(agent.request_stop())
        with WorkspaceLease(self.root):
            pass

    def test_stop_waits_for_parallel_workers_on_fatal_failure(self):
        entered = threading.Barrier(2)
        drained = threading.Event()
        seen = []
        def read(index, context):
            entered.wait(5)
            if index == 0:
                raise PersistenceError("storage failed")
            if not context.cancel.wait(5):
                raise RuntimeError("worker was not cancelled")
            drained.set()
            return "drained"
        tool = Tool("read", "", {"index": {"type": "integer"}}, read,
                    read_only=True, contextual=True)
        calls = [dict(call("read", {"index": i}), id=str(i)) for i in range(2)]
        model = Mock(complete=Mock(return_value={"role": "assistant", "tool_calls": calls}))
        class Cleanup:
            def stop(self, context, **kwargs): seen.append(drained.is_set())
        agent = Agent(model, [tool], workspace=self.root, hooks=(Cleanup(),))
        with self.assertRaises(PersistenceError):
            agent.run("task")
        self.assertEqual(seen, [True])
        self.assertEqual(inspect_run(agent.last_session_directory)["calls"]["1:1"]["status"], "completed")

    def test_protocol_errors_stop_without_executing_tools(self):
        seen = []
        class Cleanup:
            def stop(self, context, **kwargs): seen.append("stop")
        model = Mock(complete=Mock(return_value={"role": "assistant", "tool_calls": [call("action")] * 2}))
        agent = Agent(model, [self.tool], workspace=self.root, hooks=(Cleanup(),))
        with self.assertRaises(ProtocolError):
            agent.run("task")
        self.assertEqual(seen, ["stop"])
        self.assertEqual(self.effects, [])

    def test_session_end_failure_does_not_skip_other_cleanup(self):
        seen = []
        class Broken:
            def session_end(self, context, **kwargs): raise RuntimeError("end failed")
        class Later:
            def session_end(self, context, **kwargs): seen.append("end")
        agent = Agent(Model(), [], workspace=self.root, hooks=(Broken(), Later()))
        with self.assertRaisesRegex(RuntimeError, "end failed"):
            agent.run("task")
        self.assertEqual(seen, ["end"])
        self.assertEqual(inspect_run(agent.last_session_directory)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
