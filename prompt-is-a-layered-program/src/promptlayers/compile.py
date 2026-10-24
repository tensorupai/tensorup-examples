"""Compile a PromptProgram into LiteLLM completion() arguments for one model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from promptlayers.layers import PromptProgram
from promptlayers.profiles import ModelProfile


class UnsupportedLayerError(ValueError):
    """The target model cannot carry a layer the program needs. Raised, never dropped."""


VALUE_KEYWORDS = frozenset({"const", "enum", "examples"})  # hold data, not subschemas


def output_schema(output: type[BaseModel], rejected: frozenset[str]) -> dict[str, Any]:
    """Strict JSON Schema for L6, minus keywords the target model rejects.

    Covers models built from fixed fields; open-ended objects (dict fields) are refused.
    """

    def walk(node: Any, in_properties: bool = False) -> Any:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node
        if in_properties:  # keys here are field names, not keywords
            return {name: walk(sub) for name, sub in node.items()}
        out = {
            k: v if k in VALUE_KEYWORDS else walk(v, k in ("properties", "$defs"))
            for k, v in node.items()
            if k not in rejected and k != "default"  # strict mode makes every field required
        }
        if out.get("type") == "object":
            if "properties" not in out:
                raise UnsupportedLayerError("open-ended object fields cannot be strict")
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out

    return walk(output.model_json_schema())


def compile_request(
    program: PromptProgram, profile: ModelProfile, *, temperature: float | None = None
) -> dict[str, Any]:
    # L1: LiteLLM moves a "system" message to Anthropic's top-level system field.
    instructions = profile.instruction_override or program.instructions
    messages: list[dict[str, Any]] = [{"role": "system", "content": instructions}]
    for ex in program.examples:  # L3: demonstrations as prior turns
        messages.append({"role": "user", "content": ex.input})
        messages.append({"role": "assistant", "content": ex.output})
    docs = "".join(f"<document>\n{doc}\n</document>\n" for doc in program.context)
    messages.append({"role": "user", "content": f"{docs}<task>\n{program.task}\n</task>"})  # L4, L5

    request: dict[str, Any] = {"model": profile.model, "messages": messages}
    if program.tools:  # L2
        request["tools"] = list(program.tools)

    if not profile.supports_json_schema:  # L6
        raise UnsupportedLayerError(f"{profile.model} cannot enforce a JSON schema")
    request["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": program.output.__name__,
            "schema": output_schema(program.output, profile.rejected_schema_keywords),
            "strict": True,
        },
    }

    if temperature is not None:  # request config, not a layer, but it breaks on upgrades too
        if not profile.supports_temperature:
            raise UnsupportedLayerError(f"{profile.model} does not accept temperature")
        request["temperature"] = temperature
    return request
