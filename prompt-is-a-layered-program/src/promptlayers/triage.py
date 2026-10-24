"""Example program: support-ticket triage. Live: TENSORUP_LIVE=1 python -m promptlayers.triage"""

from __future__ import annotations

import os
import sys
from typing import Literal

from pydantic import BaseModel, Field

from promptlayers.layers import Example, PromptProgram
from promptlayers.profiles import profile_from_litellm


class Triage(BaseModel):
    category: Literal["billing", "bug", "account", "other"]
    priority: int = Field(ge=1, le=4)
    summary: str = Field(max_length=120)


def triage_program(ticket: str, policy_docs: tuple[str, ...] = ()) -> PromptProgram:
    return PromptProgram(
        instructions=(
            "You triage customer support tickets. Priority 1 is an outage or data loss; "
            "4 is a question. Summaries are one sentence."
        ),
        examples=(
            Example(
                input="<task>\nI was charged twice this month.\n</task>",
                output='{"category": "billing", "priority": 2, "summary": "Duplicate charge."}',
            ),
        ),
        context=policy_docs,
        task=ticket,
        output=Triage,
    )


def main() -> int:
    if os.environ.get("TENSORUP_LIVE") != "1":
        print("Set TENSORUP_LIVE=1 and OPENAI_API_KEY to call a real model.")
        return 1
    from promptlayers.run import LiteLLMModel, run

    profile = profile_from_litellm(os.environ.get("TENSORUP_MODEL", "openai/gpt-4o-2024-08-06"))
    result = run(triage_program("Exports have failed since 09:00 UTC."), profile, LiteLLMModel())
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
