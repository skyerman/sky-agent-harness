import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

from sky_agent import Agent, StepLimitExceeded
from sky_agent.tools import workspace_tools
from sky_agent.execution import ToolError
from sky_agent.persistence import inspect_run
from sky_agent.workspace import WorkspaceLease


def assistant(text=None, calls=None):
    message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = calls
    return message


def call(name, arguments, id="call_1"):
    return {"id": id, "type": "function", "function": {
        "name": name, "arguments": arguments,
    }}


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append(copy.deepcopy(messages))
        return next(self.responses)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tools = workspace_tools(self.root)

    def test_direct_answer(self):
        result = Agent(FakeModel([assistant("done")]), [], workspace=self.root).run("hello")
        self.assertEqual(result.text, "done")
        self.assertEqual(result.steps, 1)

    def test_coding_loop_preserves_multiple_call_results(self):
        model = FakeModel([
            assistant(calls=[
                call("write_file", json.dumps({"path": "hello.py", "content": "print(42)\n"}), "write"),
                call("read_file", '{"path": "hello.py"}', "read"),
            ]),
            assistant(calls=[call("run_command", json.dumps({"argv": [sys.executable, "hello.py"]}), "run")]),
            assistant("Created and verified hello.py"),
        ])
        result = Agent(model, self.tools).run("Create hello.py")
        self.assertEqual(result.steps, 3)
        self.assertEqual((self.root / "hello.py").read_text(), "print(42)\n")
        results = [m for m in model.requests[1] if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in results], ["write", "read"])
        self.assertEqual(json.loads(results[1]["content"])["result"]["text"], "print(42)\n")
        command_result = json.loads(model.requests[2][-1]["content"])["result"]
        self.assertEqual(command_result["exit_code"], 0)
        self.assertEqual(command_result["stdout"].strip(), "42")

    def test_tool_errors_are_returned_to_model(self):
        for name, args in [
            ("missing", "{}"), ("read_file", "invalid json"),
            ("read_file", "[]"), ("read_file", '{"path": "missing.txt"}'),
            ("write_file", '{"path": "../escape.txt", "content": "x"}'),
            ("run_command", '{"argv": "echo hello"}'),
        ]:
            with self.subTest(name=name, args=args):
                model = FakeModel([assistant(calls=[call(name, args)]), assistant("recovered")])
                result = Agent(model, self.tools).run("task")
                self.assertEqual(result.text, "recovered")
                self.assertFalse(json.loads(model.requests[1][-1]["content"])["ok"])

    def test_step_limit_keeps_transcript(self):
        model = FakeModel([assistant(calls=[call("read_file", '{"path": "missing"}')])])
        with self.assertRaises(StepLimitExceeded) as context:
            Agent(model, self.tools, max_steps=1).run("task")
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(context.exception.messages[-1]["role"], "tool")
        with WorkspaceLease(self.root):
            pass

    def test_runs_do_not_share_history(self):
        model = FakeModel([assistant("one"), assistant("two")])
        agent = Agent(model, self.tools)
        first = agent.run("first")
        second = agent.run("second")
        self.assertNotEqual(first.session_directory, second.session_directory)
        self.assertEqual(len(model.requests[1]), 2)
        self.assertEqual(model.requests[1][-1]["content"], "second")

    def test_cancellation_persists_results_and_releases_workspace(self):
        cancel = threading.Event()
        def observe(event):
            if event["kind"] == "started":
                cancel.set()
        model = FakeModel([assistant(calls=[call("write_file", '{"path":"never","content":"x"}')])])
        agent = Agent(model, self.tools, on_event=observe)
        with self.assertRaises(ToolError) as caught:
            agent.run("task", cancel=cancel)
        self.assertEqual(caught.exception.code, "cancelled")
        summary = inspect_run(agent.last_session_directory)
        self.assertEqual(summary["status"], "cancelled")
        self.assertEqual(summary["messages"][-1]["role"], "tool")
        self.assertFalse((self.root / "never").exists())
        with WorkspaceLease(self.root):
            pass

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            Agent(FakeModel([]), [], max_steps=0)
        with self.assertRaises(ValueError):
            Agent(FakeModel([]), [self.tools[0], self.tools[0]])
        with self.assertRaises(ValueError):
            Agent(FakeModel([]), []).run(" ")

    def test_command_failure_and_timeout(self):
        command = self.tools[-1]
        output = json.loads(command.execute(json.dumps({"argv": [sys.executable, "-c", "raise SystemExit(7)"]})))
        self.assertEqual(output["error"]["code"], "command_failed")
        self.assertEqual(output["error"]["details"]["exit_code"], 7)
        command = workspace_tools(self.root, command_timeout=0.1)[-1]
        output = json.loads(command.execute(json.dumps({"argv": [sys.executable, "-c", "import time; time.sleep(5)"]})))
        self.assertFalse(output["ok"])
        self.assertEqual(output["error"]["code"], "timeout")


if __name__ == "__main__":
    unittest.main()
