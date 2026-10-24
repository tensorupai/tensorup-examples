"""The six layers of a prompt program, kept provider-neutral."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class Example:
    """One few-shot demonstration: an input and the output we want for it."""

    input: str
    output: str


@dataclass(frozen=True)
class PromptProgram:
    """A prompt as separate layers; compile_request() decides how each one travels."""

    instructions: str  # L1: the rules for every request
    task: str  # L5: what this request asks for
    output: type[BaseModel]  # L6: the output contract
    tools: tuple[dict[str, Any], ...] = field(default=())  # L2: tool schemas
    examples: tuple[Example, ...] = field(default=())  # L3: demonstrations
    context: tuple[str, ...] = field(default=())  # L4: retrieved documents
