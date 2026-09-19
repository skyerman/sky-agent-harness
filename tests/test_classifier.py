import json
from pathlib import Path
import unittest

import httpx
from openai import OpenAI

from sky_agent.classifier import DeepSeekActionClassifier
from sky_agent.hooks import ToolRequest, conversation_snapshot
from sky_agent.tools import Tool


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.response = {"role": "assistant", "content": '{"decision":"allow","reason":"Requested local test"}'}
        self.finish_reason = "stop"
        self.timeout = False
        def respond(request):
            self.requests.append(json.loads(request.content))
            if self.timeout:
                raise httpx.ReadTimeout("test timeout", request=request)
            return httpx.Response(200, json={
                "id": "classification", "object": "chat.completion", "created": 0,
                "model": "deepseek-flash", "choices": [{
                    "index": 0, "finish_reason": self.finish_reason, "message": self.response,
                }],
            })
        client = OpenAI(api_key="test-key", base_url="https://example.invalid",
                        http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        self.addCleanup(client.close)
        self.classifier = DeepSeekActionClassifier(client, "deepseek-flash", timeout=2)
        tool = Tool("run_command", "", {}, lambda: None, workspace=Path.cwd())
        self.request = ToolRequest.create(tool, {"argv": ["python", "-m", "unittest"]}, conversation_snapshot([
            {"role": "user", "content": "Run the tests"},
            {"role": "assistant", "content": "I will run tests", "reasoning_content": "do not send this trace"},
        ]))

    def test_request_is_independent_json_classification_without_tools(self):
        result = self.classifier.classify(self.request)
        self.assertEqual(result.decision, "allow")
        payload = self.requests[0]
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(len(payload["messages"]), 2)
        evidence = json.loads(payload["messages"][1]["content"])
        self.assertEqual(evidence["arguments"]["argv"], ["python", "-m", "unittest"])
        self.assertNotIn("do not send this trace", json.dumps(payload))
        self.assertEqual(evidence["conversation"]["messages"][0]["content"], "Run the tests")

    def test_malformed_and_oversized_results_are_rejected(self):
        for content in ["not JSON", '{"decision":"allow"}', '{"decision":"maybe","reason":"unknown"}',
                        '{"decision":"allow","reason":"ok","extra":true}',
                        '{"decision":"deny","decision":"allow","reason":"duplicate"}',
                        '{"decision":"allow","reason":" "}', "x" * 8200, None]:
            with self.subTest(content=str(content)[:80]), self.assertRaises(Exception):
                self.response["content"] = content
                self.classifier.classify(self.request)

    def test_refusal_or_incomplete_response_is_not_permission(self):
        self.finish_reason = "length"
        with self.assertRaises(ValueError):
            self.classifier.classify(self.request)
        self.finish_reason = "stop"
        self.response["refusal"] = "Cannot classify"
        with self.assertRaises(ValueError):
            self.classifier.classify(self.request)

    def test_timeout_does_not_retry(self):
        from openai import APITimeoutError
        self.timeout = True
        with self.assertRaises(APITimeoutError):
            self.classifier.classify(self.request)
        self.assertEqual(len(self.requests), 1)

    def test_large_arguments_and_missing_context_ask_without_http(self):
        from dataclasses import replace
        request = replace(self.request, arguments_json=json.dumps({"code": "x" * 33000}))
        self.assertEqual(self.classifier.classify(request).decision, "ask")
        self.assertEqual(self.classifier.classify(replace(self.request, conversation_json=conversation_snapshot([]))).decision, "ask")
        self.assertEqual(self.requests, [])
