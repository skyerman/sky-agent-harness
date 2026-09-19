import argparse
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sky_agent.cli import configuration, main
from sky_agent.permissions import Classification
from sky_agent.persistence import inspect_run


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

    def test_permission_modes_and_noninteractive_denial(self):
        class WriteModel:
            def __init__(self, *args, **kwargs):
                pass
            def complete(self, messages, tools):
                if len(messages) == 2:
                    return {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "write", "type": "function", "function": {
                            "name": "write_file", "arguments": '{"path":"created","content":"text"}',
                        },
                    }]}
                return {"role": "assistant", "content": "finished"}
        for mode, exists in [("default", False), ("acceptEdits", True), ("readOnly", False)]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root, \
                    patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}, clear=True), \
                    patch("sky_agent.cli.OpenAIChatModel", WriteModel), patch("sys.stdin.isatty", return_value=False), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--workspace", root, "--permission-mode", mode, "task"]), 0)
                self.assertEqual((Path(root) / "created").exists(), exists)

    def test_auto_cli_uses_classifier_for_command(self):
        import json
        import sys
        class CommandModel:
            def __init__(self, *args, **kwargs):
                self.client = object()
            def complete(self, messages, tools):
                if len(messages) == 2:
                    return {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "command", "type": "function", "function": {
                            "name": "run_command", "arguments": json.dumps({"argv": [sys.executable, "-c", "print(42)"]}),
                        },
                    }]}
                return {"role": "assistant", "content": "finished"}
        with tempfile.TemporaryDirectory() as root, \
                patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}, clear=True), \
                patch("sky_agent.cli.OpenAIChatModel", CommandModel), \
                patch("sky_agent.cli.DeepSeekActionClassifier") as classifier, \
                patch("sys.stdin.isatty", return_value=False), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            classifier.return_value.classify.return_value = Classification("allow", "Requested test")
            self.assertEqual(main(["--workspace", root, "--permission-mode", "auto", "--classifier-model", "test-classifier", "task"]), 0)
            classifier.return_value.classify.assert_called_once()
            self.assertEqual(classifier.call_args.args[1], "test-classifier")
            session = next((Path(root) / ".sky-agent" / "runs").iterdir())
            call = inspect_run(session)["calls"]["1:command"]
            self.assertEqual(call["permissions"][-1]["source"], "classifier")
            self.assertEqual(call["result"]["result"]["stdout"].strip(), "42")

    def test_invalid_permission_flags_fail_before_network(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}, clear=True), \
                patch("sky_agent.cli.OpenAIChatModel") as model, contextlib.redirect_stderr(io.StringIO()):
            for args in [["--auto-denial-limit", "0"], ["--classifier-timeout", "nan"],
                         ["--deny-tool", "missing"], ["--permission-mode", "missing"]]:
                with self.subTest(args=args), self.assertRaises(SystemExit):
                    main([*args, "task"])
            model.assert_not_called()
