import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from sky_agent.execution import ExecutionContext, PREVIEW_LIMIT
from sky_agent.persistence import RunStore, inspect_run
from sky_agent.runtime import ToolRunner
from sky_agent.tools import workspace_tools
from test_tools import tool_call


class CommandTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = RunStore(self.root)
        self.context = ExecutionContext(self.store)
        self.runner = ToolRunner(workspace_tools(self.root), self.context)

    def run_code(self, code, **kwargs):
        messages = self.runner.run([tool_call("run_command", {
            "argv": [sys.executable, "-u", "-c", code], **kwargs,
        })], round_number=1)
        return json.loads(messages[0]["content"])

    def test_output_is_observed_before_exit_and_cwd_is_respected(self):
        directory = self.root / "child"
        directory.mkdir()
        events = []
        def observe(event):
            events.append(event["kind"])
            if event["kind"] == "output" and "ready" in event["text"]:
                (directory / "go").touch()
        self.context.callback = observe
        code = (
            "import pathlib, time, sys\n"
            "print('ready', flush=True)\n"
            "deadline = time.monotonic() + 5\n"
            "while not pathlib.Path('go').exists() and time.monotonic() < deadline: time.sleep(.01)\n"
            "assert pathlib.Path('go').exists(), 'output was not streamed'\n"
            "print('done', file=sys.stderr, flush=True)\n"
        )
        result = self.run_code(code, cwd="child")
        self.assertTrue(result["ok"], result)
        self.assertIn("ready", result["result"]["stdout"])
        self.assertIn("done", result["result"]["stderr"])
        self.assertLess(events.index("output"), events.index("finished"))

    def test_large_output_is_preserved_and_readable(self):
        result = self.run_code("import sys; sys.stdout.write('x' * 100000); sys.stderr.write('y' * 100000)")
        self.assertTrue(result["ok"], result)
        output = result["result"]
        for stream in ("stdout", "stderr"):
            self.assertEqual(len(output[stream]), PREVIEW_LIMIT)
            self.assertTrue(output["truncated"][stream])
            self.assertEqual((self.store.artifacts / output["artifacts"][stream]).stat().st_size, 100000)
        message = self.runner.run([tool_call("read_artifact", {
            "artifact_id": output["artifacts"]["stdout"], "offset": 99990, "limit": 100,
        })], round_number=2)[0]
        page = json.loads(message["content"])["result"]
        self.assertEqual(page["text"], "x" * 10)
        self.assertIsNone(page["next_offset"])

    def test_incremental_utf8_decoding(self):
        result = self.run_code("import os, time; os.write(1, bytes([0xe4])); time.sleep(.1); os.write(1, bytes([0xbd, 0xa0]))")
        self.assertEqual(result["result"]["stdout"], "\u4f60")

    def test_explicit_cancellation_and_later_call(self):
        def observe(event):
            if event["kind"] == "output":
                self.context.cancel.set()
        self.context.callback = observe
        calls = [tool_call("run_command", {"argv": [sys.executable, "-u", "-c",
                 "import time; print('ready', flush=True); time.sleep(30)"]}, "command"),
                 tool_call("write_file", {"path": "must-not-exist", "content": "x"}, "write")]
        messages = self.runner.run(calls, round_number=1)
        self.assertEqual([json.loads(m["content"])["error"]["code"] for m in messages], ["cancelled", "cancelled"])
        self.assertFalse((self.root / "must-not-exist").exists())
        self.assertEqual(inspect_run(self.store.directory)["calls"]["1:command"]["status"], "cancelled")

    def test_timeout_terminates_child_process(self):
        child = "import time; time.sleep(30)"
        result = self.run_code(
            f"import subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', {child!r}]); "
            "print(p.pid, flush=True); time.sleep(30)", timeout=1,
        )
        self.assertEqual(result["error"]["code"], "timeout", result)
        pid = int(result["error"]["details"]["stdout"].strip())
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x00100000, False, pid)
            if handle:
                try:
                    self.assertEqual(kernel.WaitForSingleObject(handle, 3000), 0)
                finally:
                    kernel.CloseHandle(handle)
        else:
            # A killed orphan can briefly be a zombie until init reaps it.
            status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
            self.assertTrue(not status.stdout.strip() or status.stdout.strip().startswith("Z"))

    def test_parent_exit_does_not_leave_pipe_holding_child(self):
        child = "import time; time.sleep(30)"
        result = self.run_code(
            f"import subprocess, sys; subprocess.Popen([sys.executable, '-c', {child!r}]); print('parent done')",
            timeout=2,
        )
        self.assertTrue(result["ok"], result)
        self.assertIn("parent done", result["result"]["stdout"])
