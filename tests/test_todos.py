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

from sky_agent import Agent, HookDecision, StepLimitExceeded
from sky_agent.cli import main, progress
from sky_agent.execution import ExecutionContext, ToolError
from sky_agent.permissions import PERMISSION_MODES, PermissionPolicy
from sky_agent.persistence import PersistenceError, RunStore, inspect_run
from sky_agent.runtime import ToolRunner
from sky_agent.tools import workspace_tools
from test_tools import tool_call


def item(identifier="inspect", status="pending", **kwargs):
    return {"id": identifier, "content": "Inspect the project", "status": status, **kwargs}


class PlanModel:
    def __init__(self, *args, **kwargs):
        pass

    def complete(self, messages, tools):
        if len(messages) == 2:
            return {"role": "assistant", "content": None, "tool_calls": [tool_call(
                "todo_write", {"expected_revision": 0, "todos": [item(status="in_progress")]})]}
        return {"role": "assistant", "content": "Work remains unfinished"}


class TodoTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = RunStore(self.root)
        self.context = ExecutionContext(self.store)
        self.tools = workspace_tools(self.root)
        self.runner = ToolRunner(self.tools, self.context)
        self.round = 0

    def invoke(self, name="todo_write", **arguments):
        self.round += 1
        result = self.runner.run([tool_call(name, arguments)], round_number=self.round)
        return json.loads(result[0]["content"])

    def events(self):
        return [json.loads(line) for line in (self.store.directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_full_plan_roundtrip_and_detached_reads(self):
        self.assertEqual(self.invoke("todo_read")["result"], {"revision": 0, "todos": []})
        original = [item(status="in_progress"), item("verify")]
        result = self.invoke(expected_revision=0, todos=original)["result"]
        self.assertEqual(result, {"revision": 1, "todos": original})
        result["todos"].clear()
        original[0]["content"] = "changed externally"
        current = self.store.todos.read()
        self.assertEqual(current["todos"][0]["content"], "Inspect the project")
        changed = [item(status="completed"), item("verify", "in_progress", content="Run tests")]
        self.assertEqual(self.invoke(expected_revision=1, todos=changed)["result"]["revision"], 2)
        saved = inspect_run(self.store.directory)
        self.assertEqual(saved["todos"], self.store.todos.read())
        self.assertEqual(len(saved["todo_history"]), 2)
        self.assertEqual(saved["todo_summary"]["unfinished_ids"], ["verify"])

    def test_invalid_payloads_never_partially_update(self):
        self.invoke(expected_revision=0, todos=[item()])
        before = self.store.todos.read()
        invalid = [[item(), item()], [item(status="in_progress"), item("two", "in_progress")],
                   [item(status="blocked")], [item(status="cancelled", reason=" ")],
                   [item(content="\t")], [item(status="unknown")], [item(extra="x")],
                   [item(identifier="bad\n")], [item(identifier="a" * 65)],
                   [item(content="x" * 501)], [item(reason="x" * 501)],
                   [item(str(i)) for i in range(51)], "bad", None]
        for todos in invalid:
            with self.subTest(todos=str(todos)[:100]):
                result = self.invoke(expected_revision=1, todos=todos)
                self.assertEqual(result["error"]["code"], "invalid_arguments")
                self.assertEqual(self.store.todos.read(), before)
        for revision in (True, 1.0, -1, "1"):
            self.assertFalse(self.invoke(expected_revision=revision, todos=[item()])["ok"])
        self.assertFalse(self.invoke(expected_revision=1, todos=[item()], change_reason="x" * 1001)["ok"])
        self.assertEqual(len(inspect_run(self.store.directory)["todo_history"]), 1)

    def test_removal_and_reopening_require_explanation(self):
        self.invoke(expected_revision=0, todos=[item()])
        denied = self.invoke(expected_revision=1, todos=[])
        self.assertEqual(denied["error"]["code"], "todo_reason_required")
        self.assertTrue(self.invoke(expected_revision=1, todos=[], change_reason="User narrowed the task")["ok"])
        self.invoke(expected_revision=2, todos=[item(status="completed")])
        self.assertFalse(self.invoke(expected_revision=3, todos=[item()])["ok"])
        self.assertTrue(self.invoke(expected_revision=3, todos=[item()], change_reason="Verification found a defect")["ok"])
        self.invoke(expected_revision=4, todos=[item(status="cancelled", reason="No longer needed")])
        self.assertFalse(self.invoke(expected_revision=5, todos=[item(status="in_progress")])["ok"])
        self.assertTrue(self.invoke(expected_revision=5, todos=[], change_reason="Archive cancelled work")["ok"])

    def test_all_statuses_and_zero_active_items_are_supported(self):
        todos = [item("p"), item("i", "in_progress"), item("c", "completed"),
                 item("b", "blocked", reason="Need input"), item("x", "cancelled", reason="Out of scope")]
        self.assertTrue(self.invoke(expected_revision=0, todos=todos)["ok"])
        todos[1]["status"] = "pending"
        self.assertTrue(self.invoke(expected_revision=1, todos=todos)["ok"])
        self.assertEqual(self.store.todos.summary()["unfinished_ids"], ["p", "i", "b"])

    def test_stale_revision_reports_current_version(self):
        self.invoke(expected_revision=0, todos=[item()])
        result = self.invoke(expected_revision=0, todos=[item("other")])
        self.assertEqual(result["error"]["code"], "todo_conflict")
        self.assertEqual(result["error"]["details"]["current_revision"], 1)
        self.assertEqual(self.invoke("todo_read")["result"]["todos"], [item()])

    def test_concurrent_writers_have_only_one_winner(self):
        barrier = threading.Barrier(2)
        def write(index):
            barrier.wait(5)
            try:
                return self.store.todos.write(self.context, 0, [item(str(index))])
            except ToolError as exc:
                return exc.code
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(write, range(2)))
        self.assertEqual(results.count("todo_conflict"), 1)
        self.assertEqual(self.store.todos.read()["revision"], 1)
        self.assertEqual(len(inspect_run(self.store.directory)["todo_history"]), 1)

    def test_journal_failure_does_not_publish_state_or_event(self):
        self.invoke(expected_revision=0, todos=[item()])
        before = self.store.todos.read()
        observed = []
        self.context.callback = lambda event: observed.append(event["kind"])
        record = self.store.record
        def fail(kind, **data):
            if kind == "todo_updated":
                raise PersistenceError("disk full")
            return record(kind, **data)
        with patch.object(self.store, "record", side_effect=fail), self.assertRaises(PersistenceError):
            self.invoke(expected_revision=1, todos=[item(status="completed")])
        self.assertEqual(self.store.todos.read(), before)
        self.assertNotIn("todo_updated", observed)
        self.assertTrue(self.context.cancel.is_set())
        self.assertEqual(inspect_run(self.store.directory)["todos"], before)

    def test_observer_sees_commit_and_cannot_mutate_or_roll_it_back(self):
        seen = []
        def observe(event):
            if event["kind"] == "todo_updated":
                seen.append(self.store.todos.read() == inspect_run(self.store.directory)["todos"])
                event["todos"].clear()
                raise RuntimeError("UI failed")
        self.context.callback = observe
        result = self.invoke(expected_revision=0, todos=[item()])
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["todos"], [item()])
        self.assertEqual(self.store.todos.read()["todos"], [item()])
        self.assertEqual(seen, [True])
        self.assertEqual(sum(e["kind"] == "todo_updated" for e in self.events()), 1)

    def test_failure_after_commit_keeps_recoverable_plan(self):
        record = self.store.record
        def fail(kind, **data):
            if kind == "finished":
                raise PersistenceError("disk full after commit")
            return record(kind, **data)
        with patch.object(self.store, "record", side_effect=fail), self.assertRaises(PersistenceError):
            self.invoke(expected_revision=0, todos=[item()])
        saved = inspect_run(self.store.directory)
        self.assertEqual(saved["todos"], {"revision": 1, "todos": [item()]})
        self.assertEqual(saved["calls"]["1:call"]["status"], "unknown")

    def test_truncated_tail_uses_last_committed_plan(self):
        self.invoke(expected_revision=0, todos=[item()])
        with (self.store.directory / "events.jsonl").open("ab") as file:
            file.write(b'{"kind":"todo_updated","revision":2')
        saved = inspect_run(self.store.directory)
        self.assertTrue(saved["incomplete_tail"])
        self.assertEqual(saved["todos"]["revision"], 1)

    def test_inspection_rejects_corrupt_plan_history(self):
        for overrides in ({"revision": 2}, {"revision": True}, {"todos": [item(), item()]},
                          {"expected_revision": 9}, {"todos": [item(status="blocked")]}):
            store = RunStore(self.root)
            event = {"expected_revision": 0, "revision": 1, "todos": [item()], "change_reason": ""}
            store.record("todo_updated", **(event | overrides))
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                inspect_run(store.directory)

    def test_default_modes_allow_state_writes_without_classifier(self):
        classifier = Mock()
        for mode in PERMISSION_MODES:
            self.runner.policy = PermissionPolicy(mode=mode, classifier=classifier)
            revision = self.store.todos.read()["revision"]
            self.assertTrue(self.invoke(expected_revision=revision, todos=[item()])["ok"])
        self.runner.policy = PermissionPolicy(mode="auto", read_only=True, classifier=classifier)
        self.assertTrue(self.invoke(expected_revision=5, todos=[item()])["ok"])
        self.assertTrue(self.invoke("todo_read")["ok"])
        self.assertFalse(self.invoke("write_file", path="no", content="no")["ok"])
        classifier.classify.assert_not_called()
        write = next(t for t in self.tools if t.name == "todo_write")
        self.assertFalse(write.read_only)

    def test_deny_and_ask_override_session_state_capability(self):
        for rules in ({"denied": {"todo_write"}}, {"ask": {"todo_write"}}):
            self.runner.policy = PermissionPolicy(mode="auto", allowed={"todo_write"}, **rules)
            self.assertFalse(self.invoke(expected_revision=0, todos=[item()])["ok"])
        approval = Mock(return_value=True)
        self.runner.policy = PermissionPolicy(read_only=True, ask={"todo_write"}, human_approve=approval)
        self.assertTrue(self.invoke(expected_revision=0, todos=[item()])["ok"])
        approval.assert_called_once()

    def test_hook_decisions_cannot_bypass_state_permissions(self):
        class Hook:
            def __init__(self, decision): self.decision = decision
            def before_tool(self, request, context): return HookDecision(self.decision, "Review")
        for decision, rules in (("allow", {"denied": {"todo_write"}}),
                                ("allow", {"ask": {"todo_write"}}), ("deny", {}), ("ask", {})):
            runner = ToolRunner(self.tools, self.context, hooks=(Hook(decision),),
                                policy=PermissionPolicy(mode="readOnly", **rules))
            result = runner.run([tool_call("todo_write", {"expected_revision": 0, "todos": [item()]})],
                                round_number=100 + self.round)
            self.round += 1
            self.assertFalse(json.loads(result[0]["content"])["ok"])
        self.assertEqual(self.store.todos.read()["revision"], 0)

    def test_todo_write_is_a_barrier_between_read_blocks(self):
        calls = [tool_call("todo_read", {}, "r1"), tool_call("todo_read", {}, "r2"),
                 tool_call("todo_write", {"expected_revision": 0, "todos": [item()]}, "w"),
                 tool_call("todo_read", {}, "r3"), tool_call("todo_read", {}, "r4")]
        self.assertEqual([len(batch) for batch in self.runner.batches(calls)], [2, 1, 2])
        results = self.runner.run(calls, round_number=1)
        self.assertEqual([json.loads(r["content"])["result"]["revision"] for r in results], [0, 0, 1, 1, 1])

    def test_reused_tools_and_agent_get_fresh_plans_per_run(self):
        agent = Agent(PlanModel(), self.tools, workspace=self.root)
        one = agent.run("task")
        two = agent.run("task")
        self.assertNotEqual(one.session_directory, two.session_directory)
        for result in (one, two):
            self.assertTrue(json.loads(result.messages[3]["content"])["ok"])
            self.assertEqual(inspect_run(result.session_directory)["todos"]["revision"], 1)
        self.assertEqual(self.store.todos.read()["revision"], 0)

    def test_stop_preserves_unfinished_items_on_all_exit_paths(self):
        for exit_kind in ("normal", "cancel", "limit", "error"):
            observed = []
            class Cleanup:
                def stop(self, context, **kwargs): observed.append(context.store.todos.read())
            model = PlanModel()
            if exit_kind == "error":
                model = Mock(complete=Mock(side_effect=[PlanModel().complete([{}, {}], []), RuntimeError("model failed")]))
            def observe(event):
                if exit_kind == "cancel" and event["kind"] == "todo_updated":
                    agent.request_stop()
            agent = Agent(model, self.tools, workspace=self.root, hooks=(Cleanup(),),
                          on_event=observe, max_steps=1 if exit_kind == "limit" else 3)
            if exit_kind == "normal":
                agent.run("task")
            else:
                with self.assertRaises({"cancel": ToolError, "limit": StepLimitExceeded, "error": RuntimeError}[exit_kind]):
                    agent.run("task")
            self.assertEqual(observed, [{"revision": 1, "todos": [item(status="in_progress")]}])
            summary = inspect_run(agent.last_session_directory)
            self.assertEqual(summary["todo_summary"]["unfinished_ids"], ["inspect"])
            final = json.loads((agent.last_session_directory / "events.jsonl").read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(final["todo_summary"], summary["todo_summary"])

    def test_precancelled_write_does_not_change_plan(self):
        self.context.cancel.set()
        self.assertEqual(self.invoke(expected_revision=0, todos=[item()])["error"]["code"], "cancelled")
        self.assertEqual(self.store.todos.read()["revision"], 0)

    def test_cli_progress_quiet_and_inspect(self):
        for quiet in (False, True):
            out, err = io.StringIO(), io.StringIO()
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}, clear=True), \
                    patch("sky_agent.cli.OpenAIChatModel", PlanModel), \
                    contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                args = ["--workspace", str(self.root), "--permission-mode", "readOnly", "task"]
                self.assertEqual(main(args + (["--quiet"] if quiet else [])), 0)
            self.assertEqual("Plan r1:" in err.getvalue(), not quiet)
            self.assertEqual("Plan saved: 1 unfinished" in err.getvalue(), not quiet)
            directory = err.getvalue().split("Session saved: ")[-1].strip()
            out = io.StringIO()
            with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(out):
                self.assertEqual(main(["--inspect", directory]), 0)
            self.assertEqual(json.loads(out.getvalue())["todos"]["revision"], 1)

    def test_progress_escapes_untrusted_terminal_control_characters(self):
        out = io.StringIO()
        with contextlib.redirect_stderr(out):
            progress({"kind": "todo_updated", "revision": 1,
                      "todos": [item(content="bad\x1b[2J\ntext\u4e2d\u6587", reason="\x1b\x9b")]})
        self.assertNotIn("\x1b", out.getvalue())
        self.assertNotIn("\x9b", out.getvalue())
        self.assertIn("\\u001b", out.getvalue())
        self.assertIn("\u4e2d\u6587", out.getvalue())

    def test_model_reads_recovers_from_conflict_and_completes_plan(self):
        actions = [("todo_read", {}),
                   ("todo_write", {"expected_revision": 0, "todos": [item(status="in_progress")]}),
                   ("todo_write", {"expected_revision": 0, "todos": [item(status="completed")]}),
                   ("todo_read", {}),
                   ("todo_write", {"expected_revision": 1, "todos": [item(status="completed")]})]
        requests = []
        class Model:
            def complete(self, messages, schemas):
                index = len(requests)
                requests.append(json.loads(json.dumps(messages)))
                if index < len(actions):
                    name, args = actions[index]
                    return {"role": "assistant", "tool_calls": [tool_call(name, args)]}
                return {"role": "assistant", "content": "done"}
        result = Agent(Model(), self.tools, workspace=self.root).run("Inspect and verify")
        self.assertEqual(result.steps, 6)
        self.assertIn("todo_read/todo_write", requests[0][0]["content"])
        self.assertEqual(json.loads(requests[3][-1]["content"])["error"]["code"], "todo_conflict")
        self.assertEqual(json.loads(requests[4][-1]["content"])["result"]["revision"], 1)
        summary = inspect_run(result.session_directory)
        self.assertEqual(summary["todos"], {"revision": 2, "todos": [item(status="completed")]})
        self.assertEqual(summary["todo_summary"]["unfinished_ids"], [])
        bare_model = Mock(complete=Mock(return_value={"role": "assistant", "content": "hello"}))
        Agent(bare_model, [], workspace=self.root).run("Hello")
        self.assertNotIn("todo_read/todo_write", bare_model.complete.call_args.args[0][0]["content"])

    def test_revision_is_rechecked_after_human_approval(self):
        def approve(request):
            self.store.todos.write(self.context, 0, [item("concurrent")])
            return True
        self.runner.policy = PermissionPolicy(ask={"todo_write"}, human_approve=approve)
        result = self.invoke(expected_revision=0, todos=[item()])
        self.assertEqual(result["error"]["code"], "todo_conflict")
        self.assertEqual(self.store.todos.read()["todos"], [item("concurrent")])


if __name__ == "__main__":
    unittest.main()
