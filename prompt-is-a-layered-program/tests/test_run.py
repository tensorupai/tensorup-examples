from types import SimpleNamespace

import pytest

from promptlayers.profiles import NON_PORTABLE_KEYWORDS, ModelProfile, profile_from_litellm
from promptlayers.run import ContractError, LiteLLMModel, ToolCallError, run
from promptlayers.triage import Triage, triage_program

VALID = '{"category": "bug", "priority": 1, "summary": "Exports failing since 09:00 UTC."}'
PROFILE = ModelProfile(model="anthropic/test-model", rejected_schema_keywords=NON_PORTABLE_KEYWORDS)


def test_happy_path_returns_validated_output(fake) -> None:
    result = run(triage_program("Exports fail."), PROFILE, fake([VALID]))
    assert isinstance(result, Triage) and result.priority == 1


def test_invalid_output_is_retried_with_the_validation_error(fake) -> None:
    model = fake(['{"category": "outage", "priority": 1, "summary": "x"}', VALID])
    result = run(triage_program("Exports fail."), PROFILE, model)
    assert result.category == "bug"
    feedback = model.requests[1]["messages"][-1]["content"]
    assert "failed validation" in feedback and "category" in feedback


def test_client_side_validation_enforces_constraints_the_wire_schema_dropped(fake) -> None:
    too_long = '{"category": "bug", "priority": 1, "summary": "%s"}' % ("x" * 200)
    out_of_range = '{"category": "bug", "priority": 9, "summary": "x"}'
    with pytest.raises(ContractError, match="2 attempts"):
        run(triage_program("Exports fail."), PROFILE, fake([too_long, out_of_range]))


def test_attempt_budget_is_respected(fake) -> None:
    model = fake(["not json"] * 5)
    with pytest.raises(ContractError):
        run(triage_program("x"), PROFILE, model, max_attempts=3)
    assert len(model.requests) == 3


def test_profile_from_litellm_reads_the_bundled_model_map() -> None:
    profile = profile_from_litellm("openai/gpt-4o-2024-08-06")
    assert profile.supports_json_schema is True
    assert profile.supports_temperature is True
    assert profile.rejected_schema_keywords == NON_PORTABLE_KEYWORDS
    overridden = profile_from_litellm("openai/gpt-4o-2024-08-06", supports_temperature=False)
    assert overridden.supports_temperature is False


def test_retry_feedback_names_fields_without_echoing_the_output(fake) -> None:
    secret = '{"category": "outage", "priority": 1, "summary": "customer card 4242"}'
    model = fake([secret, VALID])
    run(triage_program("x"), PROFILE, model)
    feedback = model.requests[1]["messages"][-1]["content"]
    assert "category:" in feedback and "4242" not in feedback


def test_tool_call_reply_is_reported_not_treated_as_bad_json(monkeypatch) -> None:
    import litellm

    call = SimpleNamespace(function=SimpleNamespace(name="lookup_order"))
    message = SimpleNamespace(tool_calls=[call], content=None)
    reply = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    monkeypatch.setattr(litellm, "completion", lambda **_: reply)
    with pytest.raises(ToolCallError, match="lookup_order"):
        run(triage_program("x"), PROFILE, LiteLLMModel())
