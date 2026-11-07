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
# The judge is a stronger model than the agent it grades, from the same provider and key.
DEFAULT_OPENAI_JUDGE = "gpt-6.1-terra"
DEFAULT_ANTHROPIC_JUDGE = "claude-sonnet-5-5"

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
    """Replays scripted assistant messages in order and records the tool names it was offered."""

    def __init__(self, replies: Sequence[Message]) -> None:
        self._replies = list(replies)
        self.offered_tools: list[str] = []

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> Any:
        return {}

    async def structured_output(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        raise NotImplementedError
        yield  # pragma: no cover

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        if not self._replies:
            raise RuntimeError("FakeModel script exhausted")
        self.offered_tools = [spec["name"] for spec in tool_specs or []]
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


def _provider_and_key() -> tuple[str, str]:
    # `or` and not a get() default: a blank line copied from .env.example sets the variable to "".
    provider = os.environ.get("TENSORUP_PROVIDER") or "openai"
    if provider not in KEY_HELP:
        raise ValueError(f"TENSORUP_PROVIDER must be openai or anthropic, got {provider!r}")
    var, where = KEY_HELP[provider]
    key = os.environ.get(var)
    if not key:
        raise MissingKeyError(f"Set {var} in the environment or .env (get one at {where}).")
    return provider, key


def _build(provider: str, key: str, model_id: str) -> Model:
    if provider == "anthropic":
        from strands.models.anthropic import AnthropicModel

        return AnthropicModel(client_args={"api_key": key}, model_id=model_id, max_tokens=2048)
    # The Responses API: Chat Completions rejects function tools with reasoning on GPT-6 models.
    from strands.models.openai_responses import OpenAIResponsesModel

    return OpenAIResponsesModel(client_args={"api_key": key}, model_id=model_id)


def build_model() -> Model:
    """The agent's model: TENSORUP_PROVIDER picks the provider, TENSORUP_MODEL the model."""
    provider, key = _provider_and_key()
    default = DEFAULT_ANTHROPIC_MODEL if provider == "anthropic" else DEFAULT_OPENAI_MODEL
    return _build(provider, key, os.environ.get("TENSORUP_MODEL") or default)


def build_judge_model() -> Model:
    """The judge's model: same provider and key, TENSORUP_JUDGE_MODEL overrides the default."""
    provider, key = _provider_and_key()
    default = DEFAULT_ANTHROPIC_JUDGE if provider == "anthropic" else DEFAULT_OPENAI_JUDGE
    return _build(provider, key, os.environ.get("TENSORUP_JUDGE_MODEL") or default)
