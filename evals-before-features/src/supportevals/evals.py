"""Run the cases through the agent, score them, and gate the release on per-evaluator pass rates."""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from strands.models import Model
from strands_evals import Case, Experiment
from strands_evals.evaluators import (
    CorrectnessEvaluator,
    Evaluator,
    FaithfulnessEvaluator,
    HelpfulnessEvaluator,
    ResponseRelevanceEvaluator,
)
from strands_evals.mappers import StrandsInMemorySessionMapper
from strands_evals.telemetry import StrandsEvalsTelemetry
from strands_evals.types import EvaluationData, EvaluationOutput
from strands_evals.types.evaluation_report import EvaluationReport
from strands_evals.types.trace import Session, ToolExecutionSpan

from .agent import build_agent
from .cases import CASES
from .support import SupportDesk

CUSTOMER_ID = "C-42"
# Live runs repeat every case: one trial per case cannot tell a regression from a coin flip.
LIVE_TRIALS = 3
# Share of runs (cases x trials) each evaluator must pass. Deterministic checks are regression
# evals: every trial of every case. Judges are non-deterministic, so they get headroom.
THRESHOLDS = {
    "ToolPolicy": 1.0,
    "CitesSource": 1.0,
    "CorrectnessEvaluator": 0.85,
    "ResponseRelevanceEvaluator": 0.85,
    "HelpfulnessEvaluator": 0.85,
    "FaithfulnessEvaluator": 0.85,
}

_telemetry: StrandsEvalsTelemetry | None = None


def _get_telemetry() -> StrandsEvalsTelemetry:
    global _telemetry
    if _telemetry is None:  # one in-memory OpenTelemetry exporter per process
        _telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
    return _telemetry


def tools_called(trajectory: Session | list[Any] | None) -> list[str]:
    if isinstance(trajectory, Session):
        return [
            span.tool_call.name
            for trace in trajectory.traces
            for span in trace.spans
            if isinstance(span, ToolExecutionSpan)
        ]
    return [str(name) for name in trajectory or []]


class ToolPolicy(Evaluator[str, str]):
    """Fails a case when a must_call tool was skipped or a must_not_call tool was called."""

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        meta = evaluation_case.metadata or {}
        called = set(tools_called(evaluation_case.actual_trajectory))
        missing = [t for t in meta.get("must_call", []) if t not in called]
        forbidden = [t for t in meta.get("must_not_call", []) if t in called]
        ok = not missing and not forbidden
        reason = "tool policy met" if ok else f"missing={missing} forbidden={forbidden}"
        return [EvaluationOutput(score=float(ok), test_pass=ok, reason=reason)]

    async def evaluate_async(self, evaluation_case: EvaluationData[str, str]) -> list:
        return self.evaluate(evaluation_case)


class CitesSource(Evaluator[str, str]):
    """The answer cites exactly the expected article, or nothing when no article applies."""

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        expected = (evaluation_case.metadata or {}).get("cite")
        cited = set(re.findall(r"\[(KB-\d+)\]", str(evaluation_case.actual_output)))
        ok = cited == ({expected} if expected else set())
        reason = f"cited={sorted(cited)} expected={expected}"
        return [EvaluationOutput(score=float(ok), test_pass=ok, reason=reason)]

    async def evaluate_async(self, evaluation_case: EvaluationData[str, str]) -> list:
        return self.evaluate(evaluation_case)


def deterministic_evaluators() -> list[Evaluator]:
    return [ToolPolicy(), CitesSource()]


def judge_evaluators(judge: Model) -> list[Evaluator]:
    return [
        CorrectnessEvaluator(model=judge),
        ResponseRelevanceEvaluator(model=judge),
        HelpfulnessEvaluator(model=judge),
        FaithfulnessEvaluator(model=judge),
    ]


def make_task(model_for: Callable[[Case], Model]) -> Callable[[Case], dict[str, Any]]:
    """One fresh desk and agent per case; the trace becomes the Session the evaluators read."""
    telemetry = _get_telemetry()

    def task(case: Case) -> dict[str, Any]:
        telemetry.in_memory_exporter.clear()
        desk = SupportDesk(customer_id=CUSTOMER_ID)
        agent = build_agent(model_for(case), desk, {"session.id": case.session_id})
        answer = str(agent(case.input))
        telemetry.tracer_provider.force_flush()
        spans = telemetry.in_memory_exporter.get_finished_spans()
        session = StrandsInMemorySessionMapper().map_to_session(spans, session_id=case.session_id)
        return {"output": answer, "trajectory": session}

    return task


def with_trials(cases: list[Case], trials: int) -> list[Case]:
    """Each case repeated `trials` times, each copy with its own session so traces stay apart."""
    if trials < 1:
        raise ValueError("trials must be at least 1")
    return [
        case.model_copy(update={"session_id": str(uuid.uuid4())})
        for case in cases
        for _ in range(trials)
    ]


def run_suite(
    task: Callable[[Case], dict[str, Any]],
    evaluators: list[Evaluator],
    cases: list[Case] | None = None,
    trials: int = 1,
) -> EvaluationReport:
    """One report for all evaluators; each row's `evaluator` field names who scored it."""
    runs = with_trials(cases or CASES, trials)
    experiment = Experiment[str, str](cases=runs, evaluators=evaluators)
    return experiment.run_evaluations(task)


def gate(report: EvaluationReport) -> list[str]:
    """Returns one line per evaluator below its threshold; an empty list means ship."""
    rows: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for case, passed in zip(report.cases, report.test_passes, strict=True):
        rows[case["evaluator"]].append((case["name"], passed))
    failures = []
    for name, results in rows.items():
        rate = sum(passed for _, passed in results) / len(results)
        if rate < THRESHOLDS[name]:
            failed = sorted({case for case, passed in results if not passed})
            failures.append(f"{name}: {rate:.0%} < {THRESHOLDS[name]:.0%}, failed {failed}")
    return failures
