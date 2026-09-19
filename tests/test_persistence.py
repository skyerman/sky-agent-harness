import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from sky_agent import Agent
from sky_agent.persistence import RunStore, inspect_run
from sky_agent.workspace import WorkspaceBusy, WorkspaceLease


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_journal_preserves_messages_and_unknown_calls(self):
        store = RunStore(self.root)
        message = {"role": "assistant", "content": None, "reasoning_content": "test provider payload"}
        store.record("message", message=message)
        store.record("started", invocation_id="1:one", tool_call_id="one", tool="write_file")
        summary = inspect_run(store.directory)
        self.assertEqual(summary["messages"], [message])
        self.assertEqual(summary["calls"]["1:one"]["status"], "unknown")
        with (store.directory / "events.jsonl").open("ab") as file:
            file.write(b'{"kind":')
        self.assertTrue(inspect_run(store.directory)["incomplete_tail"])
        with (store.directory / "events.jsonl").open("ab") as file:
            file.write(b'\n{"kind":"later"}\n')
        with self.assertRaises(ValueError):
            inspect_run(store.directory)

    def test_workspace_lease_blocks_another_process(self):
        code = (
            "from pathlib import Path\nfrom sky_agent.workspace import WorkspaceLease, WorkspaceBusy\n"
            f"try:\n with WorkspaceLease(Path({str(self.root)!r})): pass\n"
            "except WorkspaceBusy: raise SystemExit(7)\n"
        )
        with WorkspaceLease(self.root):
            process = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=10)
            self.assertEqual(process.returncode, 7, process.stderr)
        process = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_lease_is_released_on_process_death(self):
        code = (
            "from pathlib import Path\nimport os\nfrom sky_agent.workspace import WorkspaceLease\n"
            f"with WorkspaceLease(Path({str(self.root)!r})): os._exit(0)\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True, timeout=10)
        with WorkspaceLease(self.root):
            pass

    def test_model_failure_records_status_and_releases_lease(self):
        class BrokenModel:
            def complete(self, messages, tools):
                raise RuntimeError("simulated API failure")
        agent = Agent(BrokenModel(), [], workspace=self.root)
        with self.assertRaises(RuntimeError):
            agent.run("task")
        self.assertEqual(inspect_run(agent.last_session_directory)["status"], "failed")
        with WorkspaceLease(self.root):
            pass

    def test_artifact_reader_rejects_other_session(self):
        one = RunStore(self.root)
        two = RunStore(self.root)
        identifier, path = one.new_artifact("stdout")
        path.write_text("payload", encoding="utf-8")
        self.assertEqual(one.read_artifact(identifier)["text"], "payload")
        with self.assertRaises(ValueError):
            two.read_artifact(identifier)

    def test_cli_inspection_needs_no_credentials(self):
        store = RunStore(self.root)
        store.record("session_finished", status="completed")
        env = {key: value for key, value in os.environ.items() if not key.startswith(("OPENAI_", "DEEPSEEK_"))}
        process = subprocess.run([sys.executable, "-m", "sky_agent", "--inspect", str(store.directory)],
                                 env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "completed")
