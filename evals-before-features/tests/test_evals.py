import pytest
from scripts import BASELINE, UPGRADED
from strands_evals.types import EvaluationData

from supportevals.cases import CASES
from supportevals.evals import (
    THRESHOLDS,
    CitesSource,
    ToolPolicy,
    deterministic_evaluators,
    gate,
    judge_evaluators,
    make_task,
    run_suite,
    with_trials,
)
from supportevals.model import FakeModel


def scripted(script: dict):
    return make_task(lambda case: FakeModel(list(script[case.name])))


def test_every_case_has_a_baseline_script_and_complete_metadata():
    assert {c.name for c in CASES} == set(BASELINE)
    for case in CASES:
        assert {"must_call", "must_not_call", "cite"} <= set(case.metadata)


def test_baseline_model_passes_the_gate():
    report = run_suite(scripted(BASELINE), deterministic_evaluators())
    assert all(report.test_passes), report.reasons
    assert gate(report) == []


def test_upgraded_model_fails_the_gate_on_the_cases_that_drifted():
    report = run_suite(scripted(UPGRADED), deterministic_evaluators())
    failures = gate(report)
    assert len(failures) == 2
    assert failures[0].startswith("ToolPolicy")
    assert "refund-out-of-window" in failures[0] and "no-resolve-unconfirmed" in failures[0]
    assert failures[1].startswith("CitesSource") and "refund-out-of-window" in failures[1]


def test_tool_policy_reports_missing_and_forbidden_tools():
    data = EvaluationData(
        input="x",
        actual_output="done",
        actual_trajectory=["resolve_case"],
        metadata={"must_call": ["get_case"], "must_not_call": ["resolve_case"]},
    )
    [out] = ToolPolicy().evaluate(data)
    assert not out.test_pass
    assert "missing=['get_case']" in out.reason and "forbidden=['resolve_case']" in out.reason


def test_cites_source_rejects_an_invented_citation_when_none_applies():
    answer = "We price-match. [KB-104]"
    data = EvaluationData(input="x", actual_output=answer, metadata={"cite": None})
    [out] = CitesSource().evaluate(data)
    assert not out.test_pass


def test_judges_share_one_pinned_model_and_every_evaluator_has_a_threshold():
    judge = FakeModel([])
    evaluators = deterministic_evaluators() + judge_evaluators(judge)
    assert all(e.model is judge for e in evaluators[2:])
    assert {type(e).__name__ for e in evaluators} == set(THRESHOLDS)


def test_trials_repeat_each_case_and_a_drift_fails_every_trial():
    runs = with_trials(CASES[:2], 3)
    assert [c.name for c in runs] == [CASES[0].name] * 3 + [CASES[1].name] * 3
    assert len({c.session_id for c in runs}) == 6

    report = run_suite(scripted(UPGRADED), deterministic_evaluators(), trials=3)
    assert len(report.test_passes) == len(CASES) * 3 * 2
    failures = gate(report)
    assert "ToolPolicy: 71% < 100%" in failures[0]


def test_one_flaky_trial_fails_a_deterministic_gate():
    trial = {"n": 0}

    def flaky(case):
        # The second of three trials of one case drifts; the other two meet the spec.
        if case.name != "no-resolve-unconfirmed":
            return FakeModel(list(BASELINE[case.name]))
        trial["n"] += 1
        return FakeModel(list((UPGRADED if trial["n"] == 2 else BASELINE)[case.name]))

    report = run_suite(make_task(flaky), deterministic_evaluators(), trials=3)
    [failure] = gate(report)
    assert failure.startswith("ToolPolicy") and "no-resolve-unconfirmed" in failure


def test_trials_must_be_positive():
    with pytest.raises(ValueError):
        with_trials(CASES, 0)
