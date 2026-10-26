"""GoalCheck: the deterministic review step. It reads system state, never the transcript."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Outcome = Literal["pass", "fail", "stalled"]


@dataclass(frozen=True)
class Verdict:
    outcome: Outcome
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class GoalCheck:
    """Success is defined on the snapshot: the service is healthy at the target replica count."""

    target_replicas: int = 3

    def unmet(self, state: dict[str, Any]) -> list[str]:
        reasons = []
        if not state["healthy"]:
            reasons.append(f"service unhealthy with db_url={state['config']['db_url']}")
        if state["replicas"] != self.target_replicas:
            reasons.append(f"replicas={state['replicas']}, want {self.target_replicas}")
        return reasons

    def review(self, before: dict[str, Any], after: dict[str, Any]) -> Verdict:
        reasons = self.unmet(after)
        if not reasons:
            return Verdict("pass")
        if after == before:
            return Verdict("stalled", ("no state change this round", *reasons))
        return Verdict("fail", tuple(reasons))
