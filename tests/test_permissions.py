from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from sky_agent import Agent
from sky_agent.approval import TerminalApproval
from sky_agent.execution import ExecutionContext, ToolError
from sky_agent.hooks import ToolRequest, conversation_snapshot
from sky_agent.permissions import ApprovalRequest, Classification, PermissionPolicy
from sky_agent.persistence import PersistenceError, RunStore, inspect_run
from sky_agent.runtime import ToolRunner
from sky_agent.tools import Tool, workspace_tools
from test_tools import tool_call


class ScriptedClassifier:
    def __init__(self, *outcomes):
        self.outcomes = iter(outcomes)
        self.requests = []

    def classify(self, request):
        self.requests.append(request)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class PermissionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = RunStore(self.root)
        self.context = ExecutionContext(self.store)
        self.effects = []
        self.action = Tool("action", "Test action", {"value": {"type": "string"}},
                           lambda value: self.effects.append(value) or value)
        self.round = 0

    def run_action(self, runner, name="action", args=None, messages=None):
        self.round += 1
        result = runner.run([tool_call(name, args if args is not None else {"value": "original"})],
                            round_number=self.round,
                            messages=messages or [{"role": "user", "content": "Perform the requested action"}])
        return json.loads(result[0]["content"])

    def events(self, kind):
        return [event for line in (self.store.directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
                if (event := json.loads(line))["kind"] == kind]

    def test_explicit_deny_and_read_only_override_every_allow_path(self):
        classifier = ScriptedClassifier(Classification("allow", "safe"))
        for policy in [PermissionPolicy(mode="auto", denied={"action"}, allowed={"action"}, classifier=classifier),
                       PermissionPolicy(mode="auto", read_only=True, allowed={"action"}, classifier=classifier)]:
            with self.subTest(policy=policy):
                result = self.run_action(ToolRunner([self.action], self.context, policy=policy))
                self.assertEqual(result["error"]["code"], "permission_denied")
        self.assertEqual(self.effects, [])
        self.assertEqual(classifier.requests, [])

    def test_explicit_ask_beats_allow_and_classifier(self):
        classifier = ScriptedClassifier(Classification("allow", "safe"))
        runner = ToolRunner([self.action], self.context, policy=PermissionPolicy(
            mode="auto", ask={"action"}, allowed={"action"}, classifier=classifier,
            approve=lambda name, args: False,
        ))
        self.assertFalse(self.run_action(runner)["ok"])
        self.assertEqual(classifier.requests, [])
        self.assertEqual(self.effects, [])

    def test_explicit_allow_skips_classifier(self):
        classifier = ScriptedClassifier()
        runner = ToolRunner([self.action], self.context, policy=PermissionPolicy(
            mode="auto", allowed={"action"}, classifier=classifier))
        self.assertTrue(self.run_action(runner)["ok"])
        self.assertEqual(classifier.requests, [])
        inspected = inspect_run(self.store.directory)["calls"]["1:call"]["permissions"]
        self.assertEqual(inspected[-1]["source"], "allow_rule")

    def test_auto_edit_and_safe_read_paths_do_not_invoke_classifier(self):
        classifier = ScriptedClassifier()
        runner = ToolRunner(workspace_tools(self.root), self.context,
                            policy=PermissionPolicy(mode="auto", classifier=classifier))
        self.assertTrue(self.run_action(runner, "write_file", {"path": "a", "content": "text"})["ok"])
        self.assertTrue(self.run_action(runner, "read_file", {"path": "a"})["ok"])
        self.assertEqual(classifier.requests, [])
        self.assertEqual([e["source"] for e in self.events("permission_decision")], ["accept_edits", "safe_read"])

    def test_read_only_metadata_does_not_make_custom_tool_safe(self):
        self.action.read_only = True
        classifier = ScriptedClassifier(Classification("deny", "Unrecognized tool"))
        runner = ToolRunner([self.action], self.context, policy=PermissionPolicy(mode="auto", classifier=classifier))
        self.assertFalse(self.run_action(runner)["ok"])
        self.assertEqual(len(classifier.requests), 1)
        self.assertEqual(self.effects, [])

    def test_default_and_accept_edits_modes(self):
        runner = ToolRunner(workspace_tools(self.root), self.context, policy=PermissionPolicy(mode="default"))
        self.assertTrue(self.run_action(runner, "list_files", {})["ok"])
        self.assertFalse(self.run_action(runner, "write_file", {"path": "a", "content": "text"})["ok"])
        runner.policy = PermissionPolicy(mode="acceptEdits")
        self.assertTrue(self.run_action(runner, "write_file", {"path": "a", "content": "text"})["ok"])
        self.assertFalse(self.run_action(runner, "run_command", {"argv": ["never-executed"]})["ok"])

    def test_classifier_allow_ask_and_deny(self):
        classifier = ScriptedClassifier(Classification("allow", "In scope"),
                                         Classification("ask", "Needs confirmation"),
                                         Classification("deny", "Out of scope"))
        approvals = []
        runner = ToolRunner([self.action], self.context, policy=PermissionPolicy(
            mode="auto", classifier=classifier,
            human_approve=lambda request: approvals.append(request) or True))
        self.assertTrue(self.run_action(runner)["ok"])
        self.assertTrue(self.run_action(runner)["ok"])
        self.assertFalse(self.run_action(runner)["ok"])
        self.assertEqual(len(approvals), 1)
        self.assertEqual(self.effects, ["original", "original"])

    def test_denial_limit_enters_sticky_human_fallback(self):
        classifier = ScriptedClassifier(Classification("deny", "No"), Classification("deny", "Still no"))
        approvals = []
        runner = ToolRunner([self.action, *workspace_tools(self.root)], self.context, policy=PermissionPolicy(
            mode="auto", classifier=classifier, rejection_threshold=2,
            human_approve=lambda request: approvals.append(request) or True))
        self.assertFalse(self.run_action(runner)["ok"])
        self.assertTrue(self.run_action(runner, "list_files", {})["ok"])
        self.assertTrue(self.run_action(runner)["ok"])
        self.assertTrue(self.run_action(runner)["ok"])
        self.assertEqual(len(classifier.requests), 2)
        self.assertEqual([request.source for request in approvals], ["denial_limit", "denial_limit"])
        self.assertTrue(self.events("permission_decision")[-1]["human_fallback"])

    def test_classifier_allow_resets_consecutive_denials(self):
        classifier = ScriptedClassifier(Classification("deny", "No"), Classification("allow", "Yes"), Classification("deny", "No"))
        runner = ToolRunner([self.action], self.context, policy=PermissionPolicy(mode="auto", classifier=classifier, rejection_threshold=2))
        self.assertFalse(self.run_action(runner)["ok"])
        self.assertTrue(self.run_action(runner)["ok"])
        self.assertFalse(self.run_action(runner)["ok"])
        self.assertFalse(self.events("permission_decision")[-1]["human_fallback"])
        self.assertEqual(self.events("permission_decision")[-1]["consecutive_denials"], 1)

    def test_classifier_failures_require_human_approval(self):
        for outcome in [TimeoutError(), ValueError("invalid JSON"), {"decision": "allow"}]:
            with self.subTest(outcome=outcome):
                classifier = ScriptedClassifier(outcome)
                runner = ToolRunner([self.action], self.context, policy=PermissionPolicy(mode="auto", classifier=classifier))
                self.assertFalse(self.run_action(runner)["ok"])
        self.assertEqual(self.effects, [])
        approvals = []
        runner = ToolRunner([self.action], self.context, policy=PermissionPolicy(
            mode="auto", classifier=ScriptedClassifier(TimeoutError()),
            human_approve=lambda request: approvals.append(request.source) or True))
        self.assertTrue(self.run_action(runner)["ok"])
        self.assertEqual(approvals, ["classifier_error"])

    def test_approval_cannot_mutate_executed_arguments(self):
        def approve(name, arguments):
            arguments["value"] = "modified"
            return True
        runner = ToolRunner([self.action], self.context, policy=PermissionPolicy(ask={"action"}, approve=approve))
        self.assertTrue(self.run_action(runner)["ok"])
        self.assertEqual(self.effects, ["original"])
        self.assertEqual(len(self.events("permission_decision")[-1]["fingerprint"]), 64)

    def test_hook_veto_and_hook_errors_prevent_execution(self):
        class Veto:
            def before_tool(self, request, context):
                raise ToolError("hook_denied", "Custom veto")
        class Broken:
            def before_tool(self, request, context):
                raise RuntimeError("broken")
        class Invalid:
            def before_tool(self, request, context):
                return False
        for hook, code in [(Veto(), "hook_denied"), (Broken(), "hook_failed"), (Invalid(), "hook_failed")]:
            with self.subTest(hook=type(hook).__name__):
                runner = ToolRunner([self.action], self.context, hooks=(hook,))
                self.assertEqual(self.run_action(runner)["error"]["code"], code)
        self.assertEqual(self.effects, [])

    def test_permission_audit_failure_prevents_execution(self):
        original = self.store.record
        def fail(kind, **data):
            if kind == "permission_decision":
                raise PersistenceError("disk full")
            return original(kind, **data)
        runner = ToolRunner([self.action], self.context)
        with patch.object(self.store, "record", side_effect=fail), self.assertRaises(PersistenceError):
            self.run_action(runner)
        self.assertEqual(self.effects, [])

    def test_cancel_after_classifier_allow_does_not_execute(self):
        context = self.context
        class CancelClassifier:
            def classify(self, request):
                context.cancel.set()
                return Classification("allow", "Approved")
        runner = ToolRunner([self.action], self.context, policy=PermissionPolicy(mode="auto", classifier=CancelClassifier()))
        self.assertEqual(self.run_action(runner)["error"]["code"], "cancelled")
        self.assertEqual(self.effects, [])

    def test_human_prompts_are_serialized_and_waiting_calls_cancel(self):
        entered = threading.Event()
        active = 0
        peak = 0
        def approve(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            entered.set()
            request.cancel.wait(5)
            active -= 1
            return True
        self.action.read_only = True
        runner = ToolRunner([self.action], self.context, policy=PermissionPolicy(mode="default", human_approve=approve))
        calls = [tool_call("action", {"value": str(i)}, str(i)) for i in range(3)]
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(runner.run, calls, round_number=1)
            self.assertTrue(entered.wait(5))
            self.context.cancel.set()
            results = pending.result(5)
        self.assertEqual(peak, 1)
        self.assertEqual([json.loads(r["content"])["error"]["code"] for r in results], ["cancelled"] * 3)
        self.assertEqual(self.effects, [])

    def test_agent_creates_fresh_permission_state_each_run(self):
        class Model:
            def complete(self, messages, tools):
                if len(messages) == 2:
                    return {"role": "assistant", "content": None, "tool_calls": [tool_call("action", {"value": "test"})]}
                return {"role": "assistant", "content": "done"}
        classifier = ScriptedClassifier(Classification("deny", "No"), Classification("allow", "Yes"))
        policy = PermissionPolicy(mode="auto", classifier=classifier, rejection_threshold=1)
        agent = Agent(Model(), [self.action], workspace=self.root, policy=policy)
        agent.run("first task")
        agent.run("second task")
        self.assertEqual(len(classifier.requests), 2)
        self.assertEqual(self.effects, ["test"])

    def test_snapshot_omits_reasoning_and_preserves_original_task(self):
        messages = [{"role": "user", "content": "Original task"}]
        messages += [{"role": "assistant", "content": "x" * 5000, "reasoning_content": "private trace"}] * 20
        snapshot = conversation_snapshot(messages)
        self.assertNotIn("private trace", snapshot)
        self.assertIn("Original task", snapshot)
        self.assertLess(len(snapshot), 20000)
        self.assertTrue(json.loads(snapshot)["truncated"])

    def test_invalid_policy_configuration(self):
        for kwargs in [{"mode": "unknown"}, {"rejection_threshold": 0}, {"rejection_threshold": True}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PermissionPolicy(**kwargs)

    def test_terminal_approval_requires_tty_and_explicit_yes(self):
        request = ApprovalRequest(ToolRequest.create(self.action, {"value": "original"}, conversation_snapshot([])),
                                  "Needs approval", "mode", threading.Event())
        approval = TerminalApproval()
        with patch("sys.stdin.isatty", return_value=False), patch.object(approval, "_readline") as read:
            self.assertFalse(approval(request))
            read.assert_not_called()
        with patch("sys.stdin.isatty", return_value=True), contextlib.redirect_stderr(io.StringIO()):
            for answer, expected in [("yes", True), ("y", True), ("", False), ("allow", False)]:
                with patch.object(approval, "_readline", return_value=answer):
                    self.assertEqual(approval(request), expected)
            with patch.object(approval, "_readline", side_effect=EOFError):
                self.assertFalse(approval(request))

    def test_terminal_poll_can_be_cancelled_without_input(self):
        cancel = threading.Event()
        cancel.set()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(TerminalApproval._readline, cancel)
            with self.assertRaises(ToolError) as caught:
                future.result(2)
        self.assertEqual(caught.exception.code, "cancelled")
