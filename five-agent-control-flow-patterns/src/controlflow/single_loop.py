"""Pattern 1, the single loop: one Strands Agent picks every next step from its whole transcript."""

from __future__ import annotations

from strands import Agent
from strands.agent import AgentResult
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.models import Model
from strands.types import Limits

from .shop import ShopSite, make_tools

LOOP_PROMPT = "You operate a web shop with open_url and click. Finish the user's task, then stop."


def build_single_loop(
    site: ShopSite, model: Model, window_size: int = 40, pin_task: bool = False
) -> Agent:
    # Every tool result (a full page snapshot) stays in the transcript until the window trims it.
    # Without pin_first, the first message trimmed is the task itself.
    manager = SlidingWindowConversationManager(
        window_size=window_size, pin_first=1 if pin_task else None
    )
    return Agent(
        model=model,
        system_prompt=LOOP_PROMPT,
        tools=make_tools(site),
        conversation_manager=manager,
        callback_handler=None,
    )


def run_single_loop(agent: Agent, task: str, turns: int = 20) -> AgentResult:
    return agent(task, limits=Limits(turns=turns, total_tokens=200_000))
