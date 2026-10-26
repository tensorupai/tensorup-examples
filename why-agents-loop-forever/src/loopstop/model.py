"""Model access: FakeModel for tests, Strands' OpenAI and Anthropic providers for live runs."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator, Sequence
from typing import Any

from strands.models import Model
from strands.types.content import Message, Messages

DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"

KEY_HELP = {
    "openai": ("OPENAI_API_KEY", "https://platform.openai.com/api-keys"),
    "anthropic": ("ANTHROPIC_API_KEY", "https://console.anthropic.com -> API Keys"),
}


class MissingKeyError(RuntimeError):
    pass


def text(content: str) -> Message:
    return {"role": "assistant", "content": [{"text": content}]}


def call(name: str, tool_use_id: str, **tool_input: Any) -> Message:
    block = {"toolUse": {"name": name, "toolUseId": tool_use_id, "input": tool_input}}
    return {"role": "assistant", "content": [block]}


class FakeModel(Model):
    """Replays scripted assistant messages in order and records what it was sent."""

    def __init__(self, replies: Sequence[Message]) -> None:
        self._replies = list(replies)
        self.received: list[Messages] = []

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> Any:
        return {}

    async def structured_output(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        raise NotImplementedError
        yield  # pragma: no cover

    async def stream(
        self, messages: Messages, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        if not self._replies:
            raise RuntimeError("FakeModel script exhausted")
        self.received.append([dict(m) for m in messages])  # type: ignore[misc]
        reply = self._replies.pop(0)
        stop = "end_turn"
        yield {"messageStart": {"role": "assistant"}}
        for block in reply["content"]:
            if "text" in block:
                yield {"contentBlockStart": {"start": {}}}
                yield {"contentBlockDelta": {"delta": {"text": block["text"]}}}
                yield {"contentBlockStop": {}}
            if "toolUse" in block:
                stop = "tool_use"
                use = block["toolUse"]
                start = {"toolUse": {"name": use["name"], "toolUseId": use["toolUseId"]}}
                yield {"contentBlockStart": {"start": start}}
                yield {
                    "contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(use["input"])}}}
                }
                yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": stop}}
        usage = {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120}
        yield {"metadata": {"usage": usage, "metrics": {"latencyMs": 0}}}


def build_model() -> Model:
    """Pick the provider from TENSORUP_PROVIDER and the model from TENSORUP_MODEL."""
    provider = os.environ.get("TENSORUP_PROVIDER", "openai")
    if provider not in KEY_HELP:
        raise ValueError(f"TENSORUP_PROVIDER must be openai or anthropic, got {provider!r}")
    var, where = KEY_HELP[provider]
    key = os.environ.get(var)
    if not key:
        raise MissingKeyError(f"Set {var} in the environment or .env (get one at {where}).")
    if provider == "anthropic":
        from strands.models.anthropic import AnthropicModel

        model_id = os.environ.get("TENSORUP_MODEL", DEFAULT_ANTHROPIC_MODEL)
        return AnthropicModel(client_args={"api_key": key}, model_id=model_id, max_tokens=2048)
    # The Responses API: Chat Completions rejects function tools with reasoning on GPT-6 models.
    from strands.models.openai_responses import OpenAIResponsesModel

    model_id = os.environ.get("TENSORUP_MODEL", DEFAULT_OPENAI_MODEL)
    return OpenAIResponsesModel(client_args={"api_key": key}, model_id=model_id)
