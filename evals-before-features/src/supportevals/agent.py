"""The support agent: a Strands harness with its built-ins off and the SupportDesk tools on."""

from __future__ import annotations

from typing import Any

from strands import Agent
from strands.models import Model
from strands_harness import create_harness

from .support import SupportDesk, make_tools

# Written after cases.py, to make those cases pass; each rule maps to case metadata.
INSTRUCTIONS = """You are the customer support assistant for an online electronics store.
- Answer policy questions only from search_kb results, and cite the article id in square
  brackets, for example [KB-101]. If no article covers the question, say so, cite nothing,
  and offer to pass the question to a human agent.
- Call get_case before you describe or change a support case.
- Call resolve_case only when the customer says the issue is fixed. If it is not fixed,
  keep the case open and offer next steps.
- Keep answers to three sentences or fewer."""


def build_agent(
    model: Model, desk: SupportDesk, trace_attributes: dict[str, Any] | None = None
) -> Agent:
    return create_harness(
        model=model,
        instructions=INSTRUCTIONS,
        tools=make_tools(desk),
        # A support bot gets no shell, files, web, memory, sessions or skills: only its tools.
        builtin_tools=[],
        builtin_plugins=[],
        background_tasks=False,
        session=False,
        memory=False,
        skills=False,
        context_manager=False,
        caching=False,
        callback_handler=None,
        trace_attributes=trace_attributes or {},
    )
