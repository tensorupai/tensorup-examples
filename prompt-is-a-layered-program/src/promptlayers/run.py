"""Send a compiled program, validate the output contract client-side, retry with a limit."""

from __future__ import annotations

import logging
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from promptlayers.compile import compile_request
from promptlayers.layers import PromptProgram
from promptlayers.profiles import ModelProfile

log = logging.getLogger("promptlayers")
T = TypeVar("T", bound=BaseModel)


class Model(Protocol):
    def complete(self, request: dict[str, Any]) -> str: ...


class LiteLLMModel:
    """Real provider access. drop_params=False: an unsupported parameter is an error."""

    def __init__(self, timeout_s: float = 30.0, num_retries: int = 2) -> None:
        self.timeout_s = timeout_s
        self.num_retries = num_retries

    def complete(self, request: dict[str, Any]) -> str:
        import litellm

        response = litellm.completion(
            **request, drop_params=False, timeout=self.timeout_s, num_retries=self.num_retries
        )
        message = response.choices[0].message
        if message.tool_calls:  # this sample has no tool loop; say so instead of failing L6
            raise ToolCallError(
                f"model requested tools: {[c.function.name for c in message.tool_calls]}"
            )
        return message.content or ""


class ToolCallError(RuntimeError):
    """The model answered with a tool call; route it to a tool loop, not to validation."""


class ContractError(RuntimeError):
    """The model did not produce a valid L6 output within the attempt budget."""


def run(
    program: PromptProgram,
    profile: ModelProfile,
    model: Model,
    *,
    max_attempts: int = 2,
    temperature: float | None = None,
) -> BaseModel:
    request = compile_request(program, profile, temperature=temperature)
    for attempt in range(1, max_attempts + 1):
        raw = model.complete(request)
        try:
            # Full Pydantic validation, including constraints stripped from the wire schema.
            return program.output.model_validate_json(raw)
        except ValidationError as err:
            log.warning("contract violation", extra={"model": profile.model, "attempt": attempt})
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or 'output'}: {e['msg']}" for e in err.errors()
            )
            feedback = (
                f"Your output failed validation: {problems}. Return only JSON matching the schema."
            )
            request = {
                **request,
                "messages": [
                    *request["messages"],
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": feedback},
                ],
            }
    raise ContractError(f"{profile.model}: no valid output after {max_attempts} attempts")
