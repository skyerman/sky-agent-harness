from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from sky_agent.execution import ExecutionContext
from sky_agent.persistence import PersistenceError, RunStore, inspect_run
from sky_agent.runtime import ProtocolError, ToolRunner
from sky_agent.tools import Tool
from test_tools import tool_call


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = RunStore(self.root)
        self.context = ExecutionContext(self.store)

    def test_contiguous_reads_overlap_and_writes_are_barriers(self):
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        state = {"active": 0, "peak": 0, "version": 0}

        def read():
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                version = state["version"]
            barrier.wait(timeout=5)
            with lock:
                state["active"] -= 1
            return version

        def write():
            self.assertEqual(state["active"], 0)
            state["version"] += 1
            return "written"

        runner = ToolRunner([Tool("read", "", {}, read, read_only=True), Tool("write", "", {}, write)],
                            self.context, max_parallel=2)
        calls = [tool_call(name, {}, str(index)) for index, name in enumerate(["read", "read", "write", "read", "read"])]
        results = runner.run(calls, round_number=1)
        self.assertEqual([json.loads(m["content"])["result"] for m in results], [0, 0, "written", 1, 1])
        self.assertEqual([m["tool_call_id"] for m in results], ["0", "1", "2", "3", "4"])
        self.assertEqual(state["peak"], 2)

    def test_worker_bound_and_out_of_order_completion(self):
        entered = threading.Event()
        release = threading.Event()
        lock = threading.Lock()
        started = []
        finished = []

        def read(index):
            with lock:
                started.append(index)
            if index == 0:
                self.assertTrue(release.wait(5))
            elif index == 1:
                entered.set()
                self.assertTrue(release.wait(5))
            return index

        self.context.callback = lambda event: finished.append(event["tool_call_id"]) if event["kind"] == "finished" else None
        runner = ToolRunner([Tool("read", "", {"index": {"type": "integer"}}, read, read_only=True)], self.context, max_parallel=2)
        calls = [tool_call("read", {"index": i}, str(i)) for i in range(4)]
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(runner.run, calls, round_number=1)
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(set(started), {0, 1})
            finally:
                release.set()
            result = future.result(5)
        self.assertEqual([json.loads(m["content"])["result"] for m in result], [0, 1, 2, 3])

    def test_duplicate_or_malformed_calls_do_not_execute_anything(self):
        effects = []
        runner = ToolRunner([Tool("write", "", {}, lambda: effects.append(1))], self.context)
        good = tool_call("write", {})
        for bad in [good, {"id": "other", "function": {}}, "invalid"]:
            with self.subTest(bad=bad), self.assertRaises(ProtocolError):
                runner.run([good, bad], round_number=1)
        self.assertEqual(effects, [])

    def test_results_keep_order_when_second_read_finishes_first(self):
        second_finished = threading.Event()
        completed = []
        def read(index):
            if index == 0:
                self.assertTrue(second_finished.wait(5))
            return index
        def observe(event):
            if event["kind"] == "finished":
                completed.append(event["tool_call_id"])
                if event["tool_call_id"] == "second":
                    second_finished.set()
        self.context.callback = observe
        runner = ToolRunner([Tool("read", "", {"index": {"type": "integer"}}, read, read_only=True)], self.context)
        result = runner.run([tool_call("read", {"index": 0}, "first"),
                             tool_call("read", {"index": 1}, "second")], round_number=1)
        self.assertEqual(completed, ["second", "first"])
        self.assertEqual([m["tool_call_id"] for m in result], ["first", "second"])

    def test_cancelled_calls_have_results_without_effects(self):
        effects = []
        self.context.cancel.set()
        runner = ToolRunner([Tool("write", "", {}, lambda: effects.append(1))], self.context)
        result = runner.run([tool_call("write", {})], round_number=1)
        self.assertEqual(json.loads(result[0]["content"])["error"]["code"], "cancelled")
        self.assertEqual(effects, [])

    def test_callback_failure_does_not_repeat_effect(self):
        effects = []
        def broken(event):
            raise RuntimeError("broken observer")
        self.context.callback = broken
        runner = ToolRunner([Tool("write", "", {}, lambda: effects.append(1))], self.context)
        self.assertTrue(json.loads(runner.run([tool_call("write", {})], round_number=1)[0]["content"])["ok"])
        self.assertEqual(effects, [1])

    def test_storage_failure_prevents_effects_or_marks_outcome_unknown(self):
        effects = []
        runner = ToolRunner([Tool("write", "", {}, lambda: effects.append(1))], self.context)
        original = self.store.record
        def fail_finish(kind, **kwargs):
            if kind == "finished":
                raise PersistenceError("disk full")
            return original(kind, **kwargs)
        with patch.object(self.store, "record", side_effect=fail_finish), self.assertRaises(PersistenceError):
            runner.run([tool_call("write", {})], round_number=1)
        self.assertEqual(effects, [1])
        self.assertEqual(inspect_run(self.store.directory)["calls"]["1:call"]["status"], "unknown")
        self.context.cancel.clear()
        with patch.object(self.store, "record", side_effect=PersistenceError("disk full")), self.assertRaises(PersistenceError):
            runner.run([tool_call("write", {})], round_number=2)
        self.assertEqual(effects, [1])
