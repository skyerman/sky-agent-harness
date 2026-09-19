import json
from pathlib import Path
import tempfile
import unittest

import httpx
from openai import OpenAI

from sky_agent import Agent
from sky_agent.model import OpenAIChatModel
from sky_agent.tools import Tool


class ModelTests(unittest.TestCase):
    def test_reasoning_content_survives_tool_round_trip(self):
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            first = len(requests) == 1
            message = {"role": "assistant", "content": None if first else "done"}
            if first:
                message.update({
                    "reasoning_content": "test reasoning payload",
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {
                        "name": "check", "arguments": "{}",
                    }}],
                })
            return httpx.Response(200, json={
                "id": "test", "object": "chat.completion", "created": 0,
                "model": "deepseek-flash", "choices": [{
                    "index": 0, "finish_reason": "tool_calls" if first else "stop",
                    "message": message,
                }],
            })

        model = OpenAIChatModel.__new__(OpenAIChatModel)
        model.model = "deepseek-flash"
        model.client = OpenAI(
            api_key="test-key", base_url="https://example.invalid",
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        )
        self.addCleanup(model.client.close)
        with tempfile.TemporaryDirectory() as directory:
            result = Agent(model, [Tool("check", "Test tool", {}, lambda: "ok")],
                           workspace=Path(directory)).run("check")
        self.assertEqual(result.text, "done")
        self.assertEqual(len(requests), 2)
        previous = requests[1]["messages"][-2]
        self.assertEqual(previous["reasoning_content"], "test reasoning payload")
        self.assertEqual(requests[1]["messages"][-1]["tool_call_id"], "call_1")
        self.assertNotIn("reasoning_content", result.messages[-1])


if __name__ == "__main__":
    unittest.main()
