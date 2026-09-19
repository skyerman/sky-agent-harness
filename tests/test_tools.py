import json
import os
from pathlib import Path
import tempfile
import unittest

from sky_agent.execution import ExecutionContext
from sky_agent.persistence import RunStore
from sky_agent.runtime import PermissionPolicy, ToolRunner
from sky_agent.tools import TEXT_LIMIT, file_hash, workspace_tools


def tool_call(name, arguments, identifier="call"):
    return {"id": identifier, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments),
    }}


class ToolTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = RunStore(self.root)
        self.context = ExecutionContext(self.store)
        self.runner = ToolRunner(workspace_tools(self.root), self.context)
        self.round = 0

    def invoke(self, name, **arguments):
        self.round += 1
        result = self.runner.run([tool_call(name, arguments)], round_number=self.round)
        return json.loads(result[0]["content"])

    def test_create_read_edit_and_overwrite(self):
        created = self.invoke("write_file", path="a.txt", content="hello\r\n")
        self.assertTrue(created["ok"])
        self.assertEqual((self.root / "a.txt").read_bytes(), b"hello\r\n")
        read = self.invoke("read_file", path="a.txt")["result"]
        self.assertEqual(read["text"], "hello\r\n")
        changed = self.invoke("edit_file", path="a.txt", old="hello", new="world", expected_hash=read["hash"])
        self.assertTrue(changed["ok"])
        self.assertEqual((self.root / "a.txt").read_bytes(), b"world\r\n")
        failed = self.invoke("write_file", path="a.txt", content="lost")
        self.assertEqual(failed["error"]["code"], "file_exists")
        failed = self.invoke("write_file", path="a.txt", content="lost", overwrite=True)
        self.assertFalse(failed["ok"])
        result = self.invoke("write_file", path="a.txt", content="replaced", overwrite=True,
                             expected_hash=changed["result"]["hash"])
        self.assertTrue(result["ok"])

    def test_stale_hash_and_ambiguous_edit_do_not_change_file(self):
        path = self.root / "a.txt"
        path.write_bytes(b"old old")
        digest = file_hash(path.read_bytes())
        failed = self.invoke("edit_file", path="a.txt", old="old", new="new", expected_hash=digest)
        self.assertEqual(failed["error"]["code"], "edit_conflict")
        changed = self.invoke("edit_file", path="a.txt", old="old", new="new", expected_hash=digest, replace_all=True)
        self.assertEqual(changed["result"]["replacements"], 2)
        for name, arguments in [("edit_file", {"old": "new", "new": "lost"}),
                                ("write_file", {"content": "lost", "overwrite": True})]:
            with self.subTest(name=name):
                failed = self.invoke(name, path="a.txt", expected_hash=digest, **arguments)
                self.assertEqual(failed["error"]["code"], "stale_file")
                self.assertEqual(path.read_bytes(), b"new new")

    def test_read_pagination_and_long_line_are_lossless(self):
        original = "x" * (TEXT_LIMIT * 2 + 5) + "\r\nsecond\nthird\n"
        (self.root / "a.txt").write_bytes(original.encode())
        position = {"start_line": 1}
        chunks = []
        while position:
            page = self.invoke("read_file", path="a.txt", limit=1, **position)["result"]
            self.assertLessEqual(len(page["text"]), TEXT_LIMIT)
            chunks.append(page["text"])
            position = page["next"]
        self.assertEqual("".join(chunks), original)
        (self.root / "empty").touch()
        self.assertIsNone(self.invoke("read_file", path="empty")["result"]["next"])

    def test_listing_search_and_generated_files(self):
        for name, content in [("a.py", "needle\nno\nneedle"), ("b.txt", "needle"),
                              ("src/c.py", "needle"), (".venv/skip.py", "needle")]:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        (self.root / "binary").write_bytes(b"needle\0")
        page = self.invoke("list_files", pattern="*.py", limit=1)["result"]
        self.assertEqual(page["files"], ["a.py"])
        self.assertEqual(self.invoke("list_files", pattern="*.py", offset=page["next_offset"])["result"]["files"], ["src/c.py"])
        matches = self.invoke("search_text", query="needle", pattern="*.py", limit=1)["result"]
        self.assertEqual(matches["matches"][0]["line"], 1)
        self.assertEqual(self.invoke("search_text", query="needle", pattern="*.py", offset=1)["result"]["matches"][0]["line"], 3)
        matches = self.invoke("search_text", query="need.e", regex=True)["result"]
        self.assertEqual(len(matches["matches"]), 4)
        self.assertEqual(matches["skipped_files"], 1)
        self.assertFalse(self.invoke("search_text", query="[", regex=True)["ok"])

    def test_invalid_arguments_and_paths(self):
        cases = [("read_file", {"path": "x", "extra": 1}),
                 ("read_file", {"path": "x", "start_line": True}),
                 ("read_file", {"path": "x", "start_line": "1"}),
                 ("run_command", {"argv": []}),
                 ("run_command", {"argv": [3]}),
                 ("write_file", {"path": "../escape", "content": "x"}),
                 ("write_file", {"path": str(self.root.parent / "escape"), "content": "x"}),
                 ("read_file", {"path": ".sky-agent/workspace.lock"}),
                 ("read_artifact", {"artifact_id": "../events.jsonl"})]
        for name, arguments in cases:
            with self.subTest(name=name, arguments=arguments):
                self.assertFalse(self.invoke(name, **arguments)["ok"])

    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "secret"
            target.write_text("secret")
            try:
                (self.root / "link").symlink_to(target)
            except OSError:
                self.skipTest("Creating symlinks is unavailable on this host")
            self.assertEqual(self.invoke("read_file", path="link")["error"]["code"], "path_denied")

    def test_read_only_policy_blocks_writes_and_commands(self):
        self.runner.policy = PermissionPolicy(read_only=True)
        for name, args in [("write_file", {"path": "blocked", "content": "x"}),
                           ("run_command", {"argv": ["not-executed"]})]:
            result = self.invoke(name, **args)
            self.assertEqual(result["error"]["code"], "permission_denied")
        self.assertFalse((self.root / "blocked").exists())
        self.assertTrue(self.invoke("list_files")["ok"])

    def test_explicit_approval_and_denial(self):
        approvals = []
        self.runner.policy = PermissionPolicy(ask={"write_file"}, approve=lambda name, args: approvals.append(name) or False)
        self.assertFalse(self.invoke("write_file", path="a", content="x")["ok"])
        self.runner.policy.approve = lambda name, args: True
        self.assertTrue(self.invoke("write_file", path="a", content="x")["ok"])
        self.assertEqual(approvals, ["write_file"])
        self.runner.policy.denied.add("read_file")
        self.assertEqual(self.invoke("read_file", path="a")["error"]["code"], "permission_denied")

    def test_version_is_rechecked_after_approval(self):
        path = self.root / "a.txt"
        path.write_bytes(b"original")
        def approve(name, args):
            path.write_bytes(b"external edit")
            return True
        self.runner.policy = PermissionPolicy(ask={"edit_file"}, approve=approve)
        result = self.invoke("edit_file", path="a.txt", old="original", new="agent edit",
                             expected_hash=file_hash(b"original"))
        self.assertEqual(result["error"]["code"], "stale_file")
        self.assertEqual(path.read_bytes(), b"external edit")

    def test_nonfinite_numbers_and_oversized_files_are_rejected(self):
        call = tool_call("run_command", {"argv": ["not-executed"]})
        call["function"]["arguments"] = '{"argv":["not-executed"],"timeout":NaN}'
        result = self.runner.run([call], round_number=1)
        self.assertEqual(json.loads(result[0]["content"])["error"]["code"], "invalid_arguments")
        with (self.root / "large").open("wb") as file:
            file.truncate(8 * 1024 * 1024 + 1)
        self.assertEqual(self.invoke("read_file", path="large")["error"]["code"], "file_too_large")

    @unittest.skipIf(os.name == "nt", "POSIX file mode check")
    def test_edit_preserves_executable_mode(self):
        path = self.root / "script"
        path.write_bytes(b"old")
        path.chmod(0o751)
        self.invoke("edit_file", path="script", old="old", new="new", expected_hash=file_hash(b"old"))
        self.assertEqual(path.stat().st_mode & 0o777, 0o751)
