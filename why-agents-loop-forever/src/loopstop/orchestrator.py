"""The control loop: planner -> executor -> GoalCheck -> planner, with every exit bounded."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from strands import Agent
from strands.models import Model
from strands.tools.executors import SequentialToolExecutor
from strands.types import Limits

from .environment import ServiceEnv, make_tools
from .guard import NoProgressGuard
from .review import GoalCheck, Verdict

logger = logging.getLogger(__name__)

PLANNER_PROMPT = (
    "You plan operations work. Given a goal, the current system state and the reviewer's "
    "findings, write a short numbered list of concrete steps. Do not execute anything."
)
EXECUTOR_PROMPT = (
    "You execute an operations plan with the tools provided. Follow the steps, then reply "
    "with one line saying what you did."
)

RunOutcome = Literal["pass", "stalled", "budget_exhausted"]


@dataclass(frozen=True)
class Round:
    plan: str
    executor_stop: str
    guard_tripped: str | None
    verdict: Verdict


@dataclass
class RunResult:
    outcome: RunOutcome
    rounds: list[Round] = field(default_factory=list)


@dataclass
class Orchestrator:
    env: ServiceEnv
    planner_model: Model
    executor_model: Callable[[], Model]
    check: GoalCheck = field(default_factory=GoalCheck)
    max_rounds: int = 3
    executor_limits: Limits = field(default_factory=lambda: Limits(turns=8, total_tokens=40_000))

    def run(self, goal: str) -> RunResult:
        planner = Agent(
            model=self.planner_model, system_prompt=PLANNER_PROMPT, callback_handler=None
        )
        result = RunResult(outcome="budget_exhausted")
        findings = "none yet"
        for n in range(1, self.max_rounds + 1):
            before = self.env.snapshot()
            # Planner sees the goal, the state and the reviewer's findings.
            prompt = f"Goal: {goal}\nState: {before}\nReviewer findings: {findings}"
            plan = str(planner(prompt)).strip()

            # Executor sees only the plan. It is rebuilt each round, so no reviewer
            # output and no previous round's transcript ever reaches it.
            guard = NoProgressGuard(self.env.fingerprint)
            executor = Agent(
                model=self.executor_model(),
                tools=make_tools(self.env),
                system_prompt=EXECUTOR_PROMPT,
                hooks=[guard],
                # One tool at a time, so the guard reads a fingerprint no sibling call is changing.
                tool_executor=SequentialToolExecutor(),
                callback_handler=None,
            )
            run = executor(f"Plan:\n{plan}", limits=self.executor_limits)

            verdict = self.check.review(before, self.env.snapshot())
            result.rounds.append(Round(plan, str(run.stop_reason), guard.tripped, verdict))
            logger.info("round=%d executor_stop=%s verdict=%s", n, run.stop_reason, verdict)

            if verdict.outcome in ("pass", "stalled"):
                result.outcome = verdict.outcome
                return result
            findings = "; ".join(verdict.reasons)
        return result
