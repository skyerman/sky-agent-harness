from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from sky_agent import Agent, HookDecision, SubagentConfig
from sky_agent.budget import ModelBudget
from sky_agent.cli import main
from sky_agent.execution import ExecutionContext, ToolError
from sky_agent.permissions import Classification, PermissionPolicy
from sky_agent.persistence import PersistenceError, RunStore, inspect_run
from sky_agent.subagents import LinkedCancel, _WorkspaceGrant
from sky_agent.tools import Tool, workspace_tools
from sky_agent.workspace import WorkspaceBusy, WorkspaceLease
from test_tools import tool_call


def task(name="one", profile="explore", prompt="Inspect a.py"):
    return {"name": name, "profile": profile, "prompt": prompt}


def answer(content="done", calls=None):
    return {"role": "assistant", "content": content, **({"tool_calls": calls} if calls else {})}


class ScriptModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.closed = False

    def complete(self, messages, tools):
        self.requests.append(json.loads(json.dumps({"messages": messages, "tools": tools})))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response

    def close(self):
        self.closed = True


class SubagentTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "a.py").write_text("value = 42\n", encoding="utf-8")
        self.tools = workspace_tools(self.root)

    def run_children(self, factory, tasks=None, *, responses=None, tools=None, **kwargs):
        root_model = ScriptModel(responses or [answer(None, [tool_call("run_subagents", {"tasks": tasks or [task()]})]),
                                               answer("root done")])
        agent = Agent(root_model, self.tools if tools is None else tools, workspace=self.root,
                      model_factory=factory, **kwargs)
        result = agent.run("Private root context")
        delegated = json.loads(result.messages[3]["content"])
        return agent, result, delegated, root_model

    def test_real_read_history_identity_reports_and_scoped_cleanup(self):
        child = ScriptModel([answer(None, [tool_call("read_file", {"path": "a.py"})]),
                             answer("a.py:1 defines value = 42")])
        lifecycle_events = []
        class RootHook:
            def stop(self, context, **kwargs): lifecycle_events.append(("root_stop", context.agent_id))
            def subagent_start(self, context, **kwargs): lifecycle_events.append(("start", context.agent_id))
            def subagent_end(self, context, **kwargs): lifecycle_events.append(("end", context.agent_id))
        class ChildHook:
            def stop(self, context, **kwargs): lifecycle_events.append(("child_stop", context.agent_id))
        agent, result, delegated, root_model = self.run_children(lambda: child, hooks=(RootHook(),), child_hooks=(ChildHook(),))
        report = delegated["result"]["results"][0]
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["steps"], 2)
        self.assertTrue(child.closed)
        self.assertNotIn("Private root context", json.dumps(child.requests))
        self.assertEqual(child.requests[0]["messages"][-1]["content"], "Inspect a.py")
        self.assertIn("value = 42", child.requests[1]["messages"][-1]["content"])
        self.assertEqual([event[0] for event in lifecycle_events], ["start", "child_stop", "end", "root_stop"])
        saved = inspect_run(result.session_directory)
        self.assertEqual(saved["subagents"][report["agent_id"]]["status"], "completed")
        child_saved = inspect_run(Path(report["session_directory"]))
        self.assertEqual(child_saved["identity"]["parent_session_id"], saved["identity"]["session_id"])
        self.assertEqual(child_saved["identity"]["parent_invocation_id"], "1:call")
        self.assertEqual(child_saved["identity"]["agent_id"], report["agent_id"])
        self.assertTrue(Path(report["session_directory"]).is_relative_to(result.session_directory / "children"))
        artifact_path = result.session_directory / "artifacts" / report["report_artifact_id"]
        self.assertEqual(artifact_path.read_text(), report["summary"])
        self.assertIn("run_subagents", {s["function"]["name"] for s in root_model.requests[0]["tools"]})

    def test_parallel_bound_order_and_parent_write_barrier(self):
        entered = threading.Barrier(2)
        active = 0
        peak = 0
        created = 0
        lock = threading.Lock()
        def factory():
            nonlocal created
            with lock:
                index = created
                created += 1
            class Child:
                def complete(self, messages, tools):
                    nonlocal active, peak
                    with lock:
                        active += 1
                        peak = max(peak, active)
                    try:
                        if index < 2:
                            entered.wait(5)
                        self_test.assertFalse((root / "written").exists())
                        return answer(messages[-1]["content"])
                    finally:
                        with lock: active -= 1
            return Child()
        self_test, root = self, self.root
        tasks = [task(str(i), prompt=f"job {i}") for i in range(4)]
        responses = [answer(None, [tool_call("run_subagents", {"tasks": tasks}, "delegate"),
                                  tool_call("write_file", {"path": "written", "content": "after"}, "write")]), answer()]
        agent, result, delegated, _ = self.run_children(factory, responses=responses)
        self.assertEqual(peak, 2)
        self.assertEqual([r["name"] for r in delegated["result"]["results"]], [str(i) for i in range(4)])
        self.assertEqual((self.root / "written").read_text(), "after")

    def test_child_cannot_write_run_command_delegate_or_use_parent_artifacts(self):
        child = ScriptModel([answer(None, [tool_call("write_file", {"path": "bad", "content": "x"}, "write"),
                                          tool_call("run_command", {"argv": ["python"]}, "command"),
                                          tool_call("run_subagents", {"tasks": [task()]}, "nested")]), answer()])
        self.run_children(lambda: child, policy=PermissionPolicy(allowed={"write_file", "run_command", "run_subagents"}))
        names = {s["function"]["name"] for s in child.requests[0]["tools"]}
        self.assertFalse({"write_file", "run_command", "run_subagents"} & names)
        codes = [json.loads(m["content"])["error"]["code"] for m in child.requests[1]["messages"] if m["role"] == "tool"]
        self.assertEqual(codes, ["unknown_tool"] * 3)
        self.assertFalse((self.root / "bad").exists())

    def test_child_cannot_read_parent_artifact_id(self):
        class Child:
            def complete(self, messages, tools):
                if len(messages) == 2:
                    return answer(None, [tool_call("read_artifact", {"artifact_id": artifact})])
                return answer()
        class RootHook:
            def session_start(self, root_context, **kwargs):
                nonlocal artifact
                artifact, path = root_context.store.new_artifact("parent")
                path.write_text("private", encoding="utf-8")
        artifact = None
        _, result, delegated, _ = self.run_children(Child, hooks=(RootHook(),))
        saved = inspect_run(Path(delegated["result"]["results"][0]["session_directory"]))
        self.assertFalse(json.loads(saved["messages"][-2]["content"])["ok"])

    def test_child_tools_intersect_parent_registration(self):
        child = ScriptModel([answer()])
        tools = [t for t in self.tools if t.name in {"read_file", "todo_read"}]
        self.run_children(lambda: child, tools=tools)
        self.assertEqual({s["function"]["name"] for s in child.requests[0]["tools"]}, {"read_file", "todo_read"})

    def test_dispatch_approval_does_not_approve_child_calls(self):
        approvals = []
        def approve(request):
            approvals.append(request.call.tool_name)
            return request.call.tool_name == "run_subagents"
        child = ScriptModel([answer(None, [tool_call("read_file", {"path": "a.py"})]), answer()])
        self.run_children(lambda: child, policy=PermissionPolicy(mode="readOnly",
                          ask={"run_subagents", "read_file"}, human_approve=approve))
        self.assertEqual(approvals, ["run_subagents", "read_file"])
        self.assertEqual(json.loads(child.requests[1]["messages"][-1]["content"])["error"]["code"], "permission_denied")

    def test_parent_deny_and_hook_ask_cannot_be_overridden_by_child_allow(self):
        class Parent:
            def before_tool(self, request, context):
                if request.tool_name == "read_file": return HookDecision("ask", "root gate")
        class Child:
            def before_tool(self, request, context): return HookDecision("allow")
        for policy in (PermissionPolicy(denied={"read_file"}), PermissionPolicy()):
            child = ScriptModel([answer(None, [tool_call("read_file", {"path": "a.py"})]), answer()])
            self.run_children(lambda: child, policy=policy, hooks=(Parent(),), child_hooks=(Child(),))
            self.assertEqual(json.loads(child.requests[1]["messages"][-1]["content"])["error"]["code"], "permission_denied")

    def test_parent_model_gate_is_mandatory_in_children(self):
        child = ScriptModel([answer()])
        class Gate:
            def before_model(self, context, **kwargs):
                if context.agent_id != "root": raise ToolError("model_veto", "No child model request")
        _, _, delegated, _ = self.run_children(lambda: child, hooks=(Gate(),))
        self.assertEqual(delegated["result"]["results"][0]["error"], "model_veto")
        self.assertEqual(child.requests, [])

    def test_concurrent_children_use_single_human_approval_lock(self):
        ready = threading.Barrier(2)
        entered = threading.Event()
        release = threading.Event()
        active = 0
        peak = 0
        def approve(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            entered.set()
            release.wait(5)
            active -= 1
            return True
        class Child:
            def complete(self, messages, tools):
                if len(messages) == 2:
                    ready.wait(5)
                    return answer(None, [tool_call("read_file", {"path": "a.py"})])
                return answer()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.run_children, Child, [task("one"), task("two")],
                                 policy=PermissionPolicy(ask={"read_file"}, human_approve=approve))
            try:
                self.assertTrue(entered.wait(5))
            finally:
                release.set()
            _, _, delegated, _ = future.result(5)
        self.assertEqual(peak, 1)
        self.assertEqual([r["status"] for r in delegated["result"]["results"]], ["completed", "completed"])

    def test_child_failure_does_not_cancel_sibling_or_root(self):
        children = iter([ScriptModel([RuntimeError("failed")]), ScriptModel([answer("sibling done")])])
        _, result, delegated, _ = self.run_children(lambda: next(children), [task("one"), task("two")],
                                                   subagent_config=SubagentConfig(max_parallel=1))
        self.assertEqual([r["status"] for r in delegated["result"]["results"]], ["failed", "completed"])
        self.assertEqual(result.text, "root done")

    def test_budget_shared_child_step_limit_and_root_exhaustion(self):
        child = ScriptModel([answer(None, [tool_call("todo_read", {})])])
        _, _, delegated, _ = self.run_children(lambda: child, subagent_config=SubagentConfig(max_steps=1))
        self.assertEqual(delegated["result"]["results"][0]["status"], "step_limit")
        root_model = ScriptModel([answer(None, [tool_call("run_subagents", {"tasks": [task()]})]), answer()])
        agent = Agent(root_model, self.tools, workspace=self.root, model_factory=lambda: ScriptModel([answer()]), max_model_calls=1)
        with self.assertRaises(ToolError) as caught:
            agent.run("task")
        self.assertEqual(caught.exception.code, "budget_exceeded")
        saved = inspect_run(agent.last_session_directory)
        self.assertEqual(next(iter(saved["subagents"].values()))["status"], "budget_exceeded")
        self.assertEqual(len(root_model.requests), 1)
        self.assertEqual(saved["status"], "failed")

    def test_creation_limit_and_budget_reset_per_root_run(self):
        responses = [answer(None, [tool_call("run_subagents", {"tasks": [task()]})]),
                     answer(None, [tool_call("run_subagents", {"tasks": [task()]})]), answer()]
        _, _, delegated, model = self.run_children(lambda: ScriptModel([answer()]), responses=responses,
                                                  subagent_config=SubagentConfig(max_total=1))
        self.assertTrue(delegated["ok"])
        self.assertEqual(json.loads(model.requests[2]["messages"][-1]["content"])["error"]["code"], "subagent_limit")
        class Repeat:
            def complete(self, messages, tools):
                return answer(None, [tool_call("run_subagents", {"tasks": [task()]})]) if len(messages) == 2 else answer()
        agent = Agent(Repeat(), self.tools, workspace=self.root, model_factory=lambda: ScriptModel([answer()]),
                      max_model_calls=3, subagent_config=SubagentConfig(max_total=1))
        for _ in range(2):
            self.assertEqual(agent.run("task").text, "done")

    def test_parent_cancel_drains_child_before_root_stop(self):
        entered = threading.Event()
        release = threading.Event()
        child_stopped = threading.Event()
        events = []
        class Child:
            def complete(self, messages, tools):
                entered.set()
                release.wait(5)
                return answer()
        class ChildHook:
            def stop(self, context, **kwargs): child_stopped.set()
        class RootHook:
            def stop(self, context, **kwargs): events.append(child_stopped.is_set())
        model = ScriptModel([answer(None, [tool_call("run_subagents", {"tasks": [task()]})])])
        agent = Agent(model, self.tools, workspace=self.root, model_factory=Child,
                      hooks=(RootHook(),), child_hooks=(ChildHook(),))
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(agent.run, "task")
            try:
                self.assertTrue(entered.wait(5))
                self.assertTrue(agent.request_stop())
                self.assertFalse(child_stopped.is_set())
            finally:
                release.set()
            with self.assertRaises(ToolError) as caught:
                future.result(5)
        self.assertEqual(caught.exception.code, "cancelled")
        self.assertEqual(events, [True])
        saved = inspect_run(agent.last_session_directory)
        self.assertEqual(saved["status"], "cancelled")
        self.assertEqual(next(iter(saved["subagents"].values()))["status"], "cancelled")
        self.assertEqual(saved["messages"][-1]["role"], "tool")
        with WorkspaceLease(self.root): pass

    def test_large_report_is_bounded_and_accessible_to_parent(self):
        report = "finding " * 2000
        child = ScriptModel([answer(report)])
        _, result, delegated, _ = self.run_children(lambda: child)
        returned = delegated["result"]["results"][0]
        self.assertEqual(len(returned["summary"]), 4000)
        self.assertTrue(returned["truncated"])
        self.assertEqual((result.session_directory / "artifacts" / returned["report_artifact_id"]).read_text(), report)

    def test_child_todo_and_reasoning_are_isolated(self):
        class Child:
            def complete(self, messages, tools):
                if len(messages) == 2:
                    response = answer(None, [tool_call("todo_write", {"expected_revision": 0,
                        "todos": [{"id": "own", "content": "Investigate", "status": "completed"}]})])
                    response["reasoning_content"] = "child private"
                    return response
                self_test.assertEqual(messages[2]["reasoning_content"], "child private")
                return answer()
        self_test = self
        _, result, delegated, root_model = self.run_children(Child)
        saved = inspect_run(result.session_directory)
        self.assertEqual(saved["todos"], {"revision": 0, "todos": []})
        child_dir = Path(delegated["result"]["results"][0]["session_directory"])
        self.assertEqual(inspect_run(child_dir)["todos"]["revision"], 1)
        self.assertNotIn("child private", json.dumps(root_model.requests))

    def test_factory_failure_and_start_hook_veto_are_structured(self):
        factory = Mock(side_effect=RuntimeError("factory"))
        _, _, delegated, _ = self.run_children(factory)
        self.assertEqual(delegated["result"]["results"][0]["status"], "failed")
        class Gate:
            def subagent_start(self, context, **kwargs): raise ToolError("veto", "blocked")
        factory = Mock()
        _, result, delegated, _ = self.run_children(factory, hooks=(Gate(),))
        self.assertEqual(delegated["result"]["results"][0]["error"], "veto")
        self.assertIsNone(delegated["result"]["results"][0]["session_directory"])
        factory.assert_not_called()

    def test_invalid_batches_and_dispatch_rules_do_not_create_children(self):
        for tasks in ([], [task(), task()], [task(name="bad\n")], [task(prompt=" ")],
                      [task(profile="write")], [task() | {"api_key": "bad"}], [task(str(i)) for i in range(5)]):
            factory = Mock()
            _, _, delegated, _ = self.run_children(factory, responses=[answer(None, [tool_call("run_subagents", {"tasks": tasks})]), answer()])
            self.assertEqual(delegated["error"]["code"], "invalid_arguments")
            factory.assert_not_called()
        for rules in ({"denied": {"run_subagents"}}, {"ask": {"run_subagents"}}):
            factory = Mock()
            _, _, delegated, _ = self.run_children(factory, policy=PermissionPolicy(**rules))
            self.assertEqual(delegated["error"]["code"], "permission_denied")
            factory.assert_not_called()

    def test_grant_requires_live_root_lease_and_external_agents_stay_locked(self):
        with WorkspaceLease(self.root) as lease:
            grant = _WorkspaceGrant(self.root, lease)
            grant.check(self.root)
            with self.assertRaises(WorkspaceBusy):
                Agent(ScriptModel([answer()]), [], workspace=self.root).run("task")
            grant.active = False
            with self.assertRaises(RuntimeError): grant.check(self.root)
        grant.active = True
        with self.assertRaises(RuntimeError): grant.check(self.root)

    def test_budget_counts_classifier_and_has_atomic_bound(self):
        budget = ModelBudget(1)
        context = ExecutionContext(RunStore(self.root))
        permission = PermissionPolicy(mode="auto", classifier=Mock(classify=Mock(return_value=Classification("allow", "yes")))).new_session(budget=budget)
        from sky_agent.hooks import ToolRequest
        request = ToolRequest.create(Tool("custom", "", {}, lambda: None), {}, '{"messages":[]}')
        permission.before_tool(request, context)
        self.assertEqual(budget.used, 1)
        with self.assertRaises(ToolError): budget.reserve(context)

    def test_linked_cancellation_is_one_way(self):
        parent = threading.Event()
        child = LinkedCancel(parent)
        self.assertFalse(child.wait(0))
        child.set()
        self.assertFalse(parent.is_set())
        sibling = LinkedCancel(parent)
        parent.set()
        self.assertTrue(sibling.wait(1))

    def test_invalid_configuration(self):
        for kwargs in ({"max_parallel": 0}, {"max_batch": 5}, {"max_steps": True}, {"max_total": 65}):
            with self.assertRaises(ValueError): SubagentConfig(**kwargs)
        for limit in (0, True, 10001):
            with self.assertRaises(ValueError): Agent(ScriptModel([]), [], workspace=self.root, max_model_calls=limit)

    def test_fatal_child_storage_failure_wakes_other_child_and_drains(self):
        entered = threading.Barrier(2)
        seen = []
        contexts = {}
        lock = threading.Lock()
        class Hook:
            def session_start(self, context, *, task):
                with lock: contexts[task] = context
        class RootHook:
            def stop(self, context, **kwargs): seen.append(all(c.cancel.is_set() for c in contexts.values()))
        class Child:
            def complete(self, messages, tools):
                name = messages[-1]["content"]
                entered.wait(5)
                if name == "fail":
                    raise PersistenceError("failed child journal")
                if not contexts[name].cancel.wait(5):
                    raise RuntimeError("sibling was not cancelled")
                return answer()
        model = ScriptModel([answer(None, [tool_call("run_subagents", {"tasks": [task("one", prompt="wait"), task("two", prompt="fail")]})])])
        agent = Agent(model, self.tools, workspace=self.root, model_factory=Child,
                      child_hooks=(Hook(),), hooks=(RootHook(),))
        with self.assertRaises(PersistenceError): agent.run("task")
        self.assertEqual(seen, [True])
        self.assertEqual(inspect_run(agent.last_session_directory)["status"], "failed")
        with WorkspaceLease(self.root): pass

    def test_reporting_failure_is_root_fatal_and_not_successful_tool_result(self):
        model = ScriptModel([answer(None, [tool_call("run_subagents", {"tasks": [task()]})])])
        agent = Agent(model, self.tools, workspace=self.root, model_factory=lambda: ScriptModel([answer()]))
        record = RunStore.record
        def fail(store, kind, **data):
            if kind == "subagent_finished": raise PersistenceError("result unavailable")
            return record(store, kind, **data)
        with patch.object(RunStore, "record", fail), self.assertRaises(PersistenceError): agent.run("task")
        saved = inspect_run(agent.last_session_directory)
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["calls"]["1:call"]["status"], "unknown")
        self.assertEqual(next(iter(saved["subagents"].values()))["status"], "running")

    def test_child_cleanup_failure_is_local_and_root_stop_runs_once(self):
        stops = []
        class Root:
            def stop(self, context, **kwargs): stops.append("root")
        class Child:
            def stop(self, context, **kwargs):
                stops.append("child")
                raise RuntimeError("child cleanup")
        _, _, delegated, _ = self.run_children(lambda: ScriptModel([answer()]), hooks=(Root(),), child_hooks=(Child(),))
        self.assertEqual(delegated["result"]["results"][0]["status"], "failed")
        self.assertEqual(stops, ["child", "root"])

    def test_classification_counter_survives_delegation(self):
        custom = Tool("custom", "", {}, lambda: "effect", workspace=self.root)
        classifier = Mock(classify=Mock(return_value=Classification("deny", "Denied")))
        approval = Mock(return_value=False)
        policy = PermissionPolicy(mode="auto", classifier=classifier, rejection_threshold=2, human_approve=approval)
        responses = [answer(None, [tool_call("custom", {})]),
                     answer(None, [tool_call("run_subagents", {"tasks": [task()]})]),
                     answer(None, [tool_call("custom", {})]),
                     answer(None, [tool_call("custom", {})]), answer()]
        child = ScriptModel([answer(None, [tool_call("read_file", {"path": "a.py"})]), answer()])
        self.run_children(lambda: child, tools=[*self.tools, custom], policy=policy, responses=responses)
        self.assertEqual(classifier.classify.call_count, 2)
        self.assertEqual(approval.call_count, 2)

    def test_budget_concurrency_cannot_overreserve(self):
        budget = ModelBudget(2)
        context = ExecutionContext(RunStore(self.root))
        def reserve(_):
            try:
                budget.reserve(context)
                return "reserved"
            except ToolError: return "denied"
        with ThreadPoolExecutor(max_workers=4) as pool: results = list(pool.map(reserve, range(4)))
        self.assertEqual(results.count("reserved"), 2)
        self.assertEqual(budget.used, 2)

    def test_cli_factory_and_disabled_mode_and_validation(self):
        class CliModel:
            def __init__(self, *args, **kwargs): pass
            def complete(self, messages, schemas):
                if len(messages) == 2 and any(s["function"]["name"] == "run_subagents" for s in schemas):
                    return answer(None, [tool_call("run_subagents", {"tasks": [task()]})])
                return answer()
        for disabled in (False, True):
            out, err = io.StringIO(), io.StringIO()
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}, clear=True), \
                    patch("sky_agent.cli.OpenAIChatModel", side_effect=CliModel) as factory, \
                    contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.assertEqual(main(["--workspace", str(self.root), "--read-only", "task"] +
                                      (["--no-subagents"] if disabled else [])), 0)
            self.assertEqual(factory.call_count, 1 if disabled else 2)
            if not disabled:
                self.assertIn("Subagent", err.getvalue())
                directory = err.getvalue().split("Session saved: ")[-1].strip()
                saved = io.StringIO()
                with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(saved):
                    self.assertEqual(main(["--inspect", directory]), 0)
                self.assertEqual(len(json.loads(saved.getvalue())["subagents"]), 1)
        with patch("sky_agent.cli.OpenAIChatModel") as factory, contextlib.redirect_stderr(io.StringIO()):
            for args in (["--subagent-parallel", "0"], ["--subagent-max-steps", "101"],
                         ["--max-model-calls", "0"], ["--subagent-max-total", "65"]):
                with self.assertRaises(SystemExit): main([*args, "task"])
            factory.assert_not_called()

    def test_subagent_end_observer_cannot_erase_report(self):
        class Broken:
            def subagent_end(self, context, **kwargs): raise RuntimeError("observer")
        _, result, delegated, _ = self.run_children(lambda: ScriptModel([answer("finding")]), hooks=(Broken(),))
        self.assertEqual(delegated["result"]["results"][0]["summary"], "finding")
        self.assertTrue(any(e.get("method") == "subagent_end" and e["kind"] == "hook_failed"
                            for e in inspect_run(result.session_directory)["hooks"]))

    def test_storage_failure_during_child_cleanup_is_fatal_even_with_model_error(self):
        class Cleanup:
            def stop(self, context, **kwargs): raise PersistenceError("cleanup disk failure")
        model = ScriptModel([answer(None, [tool_call("run_subagents", {"tasks": [task()]})])])
        agent = Agent(model, self.tools, workspace=self.root,
                      model_factory=lambda: ScriptModel([RuntimeError("model error")]), child_hooks=(Cleanup(),))
        with self.assertRaises(PersistenceError): agent.run("task")
        self.assertEqual(inspect_run(agent.last_session_directory)["status"], "failed")

    def test_deepseek_http_child_reasoning_survives_without_leaking_to_parent(self):
        import httpx
        from openai import OpenAI
        from sky_agent.model import OpenAIChatModel
        requests = []
        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            names = {tool["function"]["name"] for tool in payload["tools"]}
            is_root = "run_subagents" in names
            initial = len(payload["messages"]) == 2
            if initial:
                message = answer(None, [tool_call("run_subagents", {"tasks": [task()]})] if is_root else
                                 [tool_call("read_file", {"path": "a.py"})])
                message["reasoning_content"] = "root trace" if is_root else "child trace"
            else:
                message = answer("root summary" if is_root else "a.py:1 verified")
            return httpx.Response(200, json={"id": "mock", "object": "chat.completion", "created": 0,
                "model": "deepseek-flash", "choices": [{"index": 0, "message": message,
                "finish_reason": "tool_calls" if initial else "stop"}]})
        def factory():
            model = OpenAIChatModel.__new__(OpenAIChatModel)
            model.model = "deepseek-flash"
            model.client = OpenAI(api_key="test", base_url="https://example.invalid", max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(respond)))
            return model
        model = factory()
        self.addCleanup(model.close)
        result = Agent(model, self.tools, workspace=self.root, model_factory=factory).run("root task")
        self.assertEqual(result.text, "root summary")
        self.assertEqual(len(requests), 4)
        child_requests = [p for p in requests if not any(t["function"]["name"] == "run_subagents" for t in p["tools"])]
        self.assertEqual(child_requests[1]["messages"][2]["reasoning_content"], "child trace")
        self.assertEqual(requests[-1]["messages"][2]["reasoning_content"], "root trace")
        self.assertNotIn("child trace", json.dumps(requests[-1]))

    def test_keyboard_interrupt_drains_children_before_parent_cleanup(self):
        stopped = []
        class ChildHook:
            def stop(self, context, **kwargs): stopped.append("child")
        class RootHook:
            def stop(self, context, **kwargs): stopped.append("root")
        model = ScriptModel([answer(None, [tool_call("run_subagents", {"tasks": [task()]})])])
        agent = Agent(model, self.tools, workspace=self.root, model_factory=lambda: ScriptModel([KeyboardInterrupt()]),
                      child_hooks=(ChildHook(),), hooks=(RootHook(),))
        with self.assertRaises(ToolError) as caught: agent.run("task")
        self.assertEqual(caught.exception.code, "cancelled")
        self.assertEqual(stopped, ["child", "root"])

    def test_factory_cannot_reuse_root_or_previous_child_model(self):
        root_model = ScriptModel([answer(None, [tool_call("run_subagents", {"tasks": [task()]})]), answer()])
        result = Agent(root_model, self.tools, workspace=self.root, model_factory=lambda: root_model).run("task")
        report = json.loads(result.messages[3]["content"])["result"]["results"][0]
        self.assertEqual(report["error"], "ValueError")
        self.assertFalse(root_model.closed)
        child = ScriptModel([answer()])
        _, _, delegated, _ = self.run_children(lambda: child, [task("one"), task("two")],
                                               subagent_config=SubagentConfig(max_parallel=1))
        self.assertEqual([r["status"] for r in delegated["result"]["results"]], ["completed", "failed"])

    def test_model_close_failure_is_recorded_before_child_terminal_status(self):
        child = ScriptModel([answer()])
        child.close = Mock(side_effect=RuntimeError("close failure"))
        _, _, delegated, _ = self.run_children(lambda: child)
        report = delegated["result"]["results"][0]
        self.assertEqual(report["status"], "failed")
        self.assertEqual(inspect_run(Path(report["session_directory"]))["status"], "failed")
        child.close.assert_called_once()

    def test_child_preserves_parent_tool_schema_and_path_validator(self):
        read = next(t for t in self.tools if t.name == "read_file")
        def deny_path(**kwargs):
            raise ToolError("parent_path_denied", "Parent restricts this file")
        read.validator = deny_path
        child = ScriptModel([answer(None, [tool_call("read_file", {"path": "a.py"})]), answer()])
        self.run_children(lambda: child)
        self.assertEqual(json.loads(child.requests[1]["messages"][-1]["content"])["error"]["code"], "parent_path_denied")

    def test_child_store_creation_failure_is_root_fatal(self):
        model = ScriptModel([answer(None, [tool_call("run_subagents", {"tasks": [task()]})])])
        factory = Mock()
        agent = Agent(model, self.tools, workspace=self.root, model_factory=factory)
        with patch("sky_agent.subagents.RunStore", side_effect=OSError("cannot mkdir")), self.assertRaises(PersistenceError):
            agent.run("task")
        factory.assert_not_called()
        self.assertEqual(inspect_run(agent.last_session_directory)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
