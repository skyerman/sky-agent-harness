from typing import Any


class OpenAIChatModel:
    """Adapter for OpenAI-compatible Chat Completions endpoints."""

    def __init__(self, model: str, *, api_key: str, base_url: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=60, max_retries=2)
        self.model = model

    def complete(self, messages: list[dict[str, Any]], tools: list[dict]) -> dict:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        completion = self.client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        if choice.finish_reason not in {"stop", "tool_calls"}:
            raise RuntimeError(f"Model response interrupted: {choice.finish_reason}")
        message = choice.message
        if message.refusal:
            raise RuntimeError(f"Model refused the request: {message.refusal}")
        response: dict[str, Any] = {"role": "assistant", "content": message.content}
        # DeepSeek requires this field on subsequent requests with tools enabled.
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content is not None:
            response["reasoning_content"] = reasoning_content
        if message.tool_calls:
            response["tool_calls"] = [call.model_dump(exclude_none=True) for call in message.tool_calls]
        return response
