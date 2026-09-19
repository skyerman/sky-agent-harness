import argparse
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sky_agent.cli import configuration, main


class CliTests(unittest.TestCase):
    def test_deepseek_and_legacy_configuration(self):
        args = argparse.Namespace(model=None, base_url=None)
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "ds-test"}, clear=True):
            self.assertEqual(configuration(args), ("ds-test", "deepseek-flash", "https://api.deepseek.com"))
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "ds-test", "OPENAI_MODEL": "unrelated", "OPENAI_BASE_URL": "https://unrelated.invalid"}, clear=True):
            self.assertEqual(configuration(args), ("ds-test", "deepseek-flash", "https://api.deepseek.com"))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "legacy", "OPENAI_MODEL": "legacy-model", "OPENAI_BASE_URL": "https://legacy.invalid"}, clear=True):
            self.assertEqual(configuration(args), ("legacy", "legacy-model", "https://legacy.invalid"))
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "ds", "DEEPSEEK_MODEL": "ds-model"}, clear=True):
            args.model = "explicit"
            args.base_url = "https://explicit.invalid"
            self.assertEqual(configuration(args), ("ds", "explicit", "https://explicit.invalid"))

    def test_offline_cli_run(self):
        class FakeModel:
            def __init__(self, *args, **kwargs):
                self.calls = 0
            def complete(self, messages, tools):
                self.calls += 1
                if self.calls == 1:
                    return {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "create", "type": "function", "function": {
                            "name": "write_file", "arguments": '{"path":"hello.py","content":"print(42)"}',
                        },
                    }]}
                return {"role": "assistant", "content": "Created hello.py"}
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}, clear=True), \
                patch("sky_agent.cli.OpenAIChatModel", FakeModel), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            result = main(["--workspace", root, "create hello.py"])
            self.assertEqual(result, 0)
            self.assertEqual((Path(root) / "hello.py").read_text(), "print(42)")
        self.assertEqual(out.getvalue().strip(), "Created hello.py")
        self.assertIn("write_file started", err.getvalue())
        self.assertIn("Session saved:", err.getvalue())

    def test_invalid_limits_fail_before_model_creation(self):
        with patch("sky_agent.cli.OpenAIChatModel") as model, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["--max-parallel", "0", "task"])
            model.assert_not_called()
