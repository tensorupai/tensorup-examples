"""Pattern 4, planner/executor: the planner keeps plan and progress; a throwaway executor acts."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from strands import Agent
from strands.models import Model
from strands.types import Limits

from .shop import ShopSite, make_tools

logger = logging.getLogger(__name__)

PLANNER_PROMPT = (
    "You plan a web-shop task and track its progress. You never act yourself. "
    "Reply with one line only: 'NEXT: <one step for the executor>' or 'DONE: <summary>'."
)
EXECUTOR_PROMPT = (
    "You perform exactly one step in a web shop with open_url and click, "
    "then report in one sentence what changed."
)


@dataclass(frozen=True)
class StepReport:
    step: str
    report: str
    stop_reason: str


@dataclass
class PlanRun:
    outcome: Literal["done", "step_budget", "bad_reply"]
    steps: list[StepReport] = field(default_factory=list)
    summary: str = ""


class PlannerExecutor:
    def __init__(
        self,
        site: ShopSite,
        planner_model: Model,
        executor_model: Callable[[], Model],
        max_steps: int = 8,
        report_chars: int = 300,
    ) -> None:
        self.site = site
        self.planner = Agent(
            model=planner_model, system_prompt=PLANNER_PROMPT, callback_handler=None
        )
        self.executor_model = executor_model
        self.max_steps = max_steps
        self.report_chars = report_chars

    def run_step(self, step: str) -> StepReport:
        # A fresh Agent per step: its context starts empty and is thrown away with its snapshots.
        executor = Agent(
            model=self.executor_model(),
            system_prompt=EXECUTOR_PROMPT,
            tools=make_tools(self.site),
            callback_handler=None,
        )
        result = executor(
            f"Step: {step}\nCurrent page: {self.site.page}",
            limits=Limits(turns=6, total_tokens=40_000),
        )
        # Only a short report crosses back; a snapshot pasted into the reply is cut here.
        report = str(result).strip()[: self.report_chars]
        return StepReport(step, report, result.stop_reason)

    def run(self, task: str) -> PlanRun:
        run = PlanRun("step_budget")
        message = f"Task: {task}\nCurrent page: {self.site.page}"
        for _ in range(self.max_steps):
            reply = str(self.planner(message)).strip()
            # "done" is the planner's claim; check the system state before trusting it.
            if reply.startswith("DONE:"):
                run.outcome, run.summary = "done", reply.removeprefix("DONE:").strip()
                return run
            if not reply.startswith("NEXT:"):
                logger.warning("planner reply not in protocol: %r", reply[:80])
                run.outcome = "bad_reply"
                return run
            step = self.run_step(reply.removeprefix("NEXT:").strip())
            run.steps.append(step)
            logger.info("step=%r stop=%s", step.step, step.stop_reason)
            message = f"Step result ({step.stop_reason}): {step.report}\nPage: {self.site.page}"
        return run
