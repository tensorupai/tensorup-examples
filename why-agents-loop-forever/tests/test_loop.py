from strands.types import Limits

from loopstop.environment import GOOD_DB_URL, ServiceEnv
from loopstop.model import FakeModel, call, text
from loopstop.orchestrator import Orchestrator

GOAL = "Get the orders service healthy and running 3 replicas."


def make(env, planner_replies, executor_replies, **kwargs):
    planner = FakeModel(planner_replies)
    executor = FakeModel(executor_replies)
    orch = Orchestrator(env, planner, lambda: executor, **kwargs)
    return orch, planner, executor


def all_text(messages_seen) -> str:
    return " ".join(str(m) for messages in messages_seen for m in messages)


def test_happy_path_passes_in_one_round():
    env = ServiceEnv()
    orch, _, _ = make(
        env,
        [text("1. Check status. 2. Fix db_url. 3. Recheck. 4. Scale to 3.")],
        [
            call("get_status", "t1"),
            call("set_config", "t2", key="db_url", value=GOOD_DB_URL),
            call("get_status", "t3"),  # same call, new state: allowed
            call("scale", "t4", replicas=3),
            text("Fixed db_url and scaled to 3."),
        ],
    )
    result = orch.run(GOAL)
    assert result.outcome == "pass"
    assert len(result.rounds) == 1
    assert result.rounds[0].guard_tripped is None


def test_repeat_from_unchanged_state_stops_executor_and_run_stalls():
    env = ServiceEnv()
    orch, _, executor = make(
        env,
        [text("1. Restart the service.")],
        # The model would restart forever; the second identical call is cancelled.
        [call("restart_service", "t1"), call("restart_service", "t2"), text("unused")],
    )
    result = orch.run(GOAL)
    first = result.rounds[0]
    assert first.guard_tripped == "restart_service({}) repeated with no state change"
    assert first.executor_stop == "cancelled"
    assert first.verdict.outcome == "stalled"
    assert result.outcome == "stalled"
    assert env.restarts == 1
    assert len(executor.received) == 2  # no model call after the cancel


def test_verdict_reaches_planner_and_never_executor():
    env = ServiceEnv()
    orch, planner, executor = make(
        env,
        [text("1. Scale to 3."), text("1. Point db_url at db-new.")],
        [
            call("scale", "t1", replicas=3),
            text("Scaled."),
            call("set_config", "t2", key="db_url", value=GOOD_DB_URL),
            text("Updated db_url."),
        ],
    )
    result = orch.run(GOAL)
    assert [r.verdict.outcome for r in result.rounds] == ["fail", "pass"]
    assert "service unhealthy" in all_text(planner.received[1:])
    # Each executor run starts from exactly one message: the plan.
    round_two_start = executor.received[2]
    assert len(round_two_start) == 1
    assert round_two_start[0]["content"] == [{"text": "Plan:\n1. Point db_url at db-new."}]


def test_turn_limit_bounds_one_executor_run():
    env = ServiceEnv()
    orch, _, executor = make(
        env,
        [text("1. Find the right replica count.")],
        [call("scale", f"t{i}", replicas=i) for i in range(2, 10)],
        max_rounds=1,
        executor_limits=Limits(turns=2),
    )
    result = orch.run(GOAL)
    assert result.rounds[0].executor_stop == "limit_turns"
    assert result.rounds[0].verdict.outcome == "fail"
    assert result.outcome == "budget_exhausted"
    assert env.replicas == 3  # two turns ran: scale(2), scale(3)
    assert len(executor.received) == 2  # the limit stopped it, not the script


def test_token_limit_bounds_one_executor_run():
    env = ServiceEnv()
    orch, _, executor = make(
        env,
        [text("1. Find the right replica count.")],
        [call("scale", f"t{i}", replicas=i) for i in range(2, 10)],
        max_rounds=1,
        executor_limits=Limits(total_tokens=200),  # FakeModel reports 120 tokens per call
    )
    result = orch.run(GOAL)
    assert result.rounds[0].executor_stop == "limit_total_tokens"
    assert len(executor.received) == 2


def test_claimed_success_in_transcript_does_not_pass():
    env = ServiceEnv()
    orch, _, _ = make(
        env,
        [text("1. Make it healthy.")],
        [text("The service is healthy and running 3 replicas.")],
    )
    result = orch.run(GOAL)
    assert result.outcome == "stalled"
    assert "no state change this round" in result.rounds[0].verdict.reasons


def test_round_budget_ends_a_run_that_keeps_failing():
    env = ServiceEnv()
    orch, _, _ = make(
        env,
        [text("1. Scale."), text("1. Scale again.")],
        [call("scale", "t1", replicas=2), text("ok"), call("scale", "t2", replicas=4), text("ok")],
        max_rounds=2,
    )
    result = orch.run(GOAL)
    assert result.outcome == "budget_exhausted"
    assert [r.verdict.outcome for r in result.rounds] == ["fail", "fail"]
