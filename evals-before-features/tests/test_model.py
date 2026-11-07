import pytest
from strands.models.anthropic import AnthropicModel
from strands.models.openai_responses import OpenAIResponsesModel

from supportevals import __main__ as cli
from supportevals.model import (
    DEFAULT_ANTHROPIC_JUDGE,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_JUDGE,
    DEFAULT_OPENAI_MODEL,
    MissingKeyError,
    build_judge_model,
    build_model,
)
from supportevals.support import SupportDesk, make_tools

ENV = ("TENSORUP_PROVIDER", "TENSORUP_MODEL", "TENSORUP_JUDGE_MODEL", "TENSORUP_LIVE")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (*ENV, "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_blank_variables_from_env_example_fall_back_to_defaults(monkeypatch):
    for name in ENV[:3]:
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert isinstance(build_model(), OpenAIResponsesModel)
    assert build_model().get_config()["model_id"] == DEFAULT_OPENAI_MODEL
    assert build_judge_model().get_config()["model_id"] == DEFAULT_OPENAI_JUDGE


def test_anthropic_provider_and_defaults(monkeypatch):
    monkeypatch.setenv("TENSORUP_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert isinstance(build_model(), AnthropicModel)
    assert build_model().get_config()["model_id"] == DEFAULT_ANTHROPIC_MODEL
    assert build_judge_model().get_config()["model_id"] == DEFAULT_ANTHROPIC_JUDGE


def test_missing_key_and_bad_provider_raise_readable_errors(monkeypatch):
    with pytest.raises(MissingKeyError, match="OPENAI_API_KEY"):
        build_model()
    monkeypatch.setenv("TENSORUP_PROVIDER", "gemini")
    with pytest.raises(ValueError, match="openai or anthropic"):
        build_model()


def test_cli_refuses_without_live_flag_and_prints_missing_key(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["supportevals", "eval"])
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    assert cli.main() == 2
    assert "TENSORUP_LIVE=1" in capsys.readouterr().out
    monkeypatch.setenv("TENSORUP_LIVE", "1")
    assert cli.main() == 2
    assert "Set OPENAI_API_KEY" in capsys.readouterr().out


def test_openai_request_uses_responses_api_with_the_tools(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model = build_model()
    specs = [t.tool_spec for t in make_tools(SupportDesk(customer_id="C-42"))]
    messages = [{"role": "user", "content": [{"text": "hi"}]}]
    request = model._format_request(messages, specs, "system")
    assert request["model"] == DEFAULT_OPENAI_MODEL
    assert sorted(t["name"] for t in request["tools"]) == ["get_case", "resolve_case", "search_kb"]
    assert "reasoning_effort" not in request and "messages" not in request


def test_anthropic_request_carries_the_tools(monkeypatch):
    monkeypatch.setenv("TENSORUP_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    specs = [t.tool_spec for t in make_tools(SupportDesk(customer_id="C-42"))]
    messages = [{"role": "user", "content": [{"text": "hi"}]}]
    request = build_model().format_request(messages, specs, "system")
    assert request["model"] == DEFAULT_ANTHROPIC_MODEL
    assert sorted(t["name"] for t in request["tools"]) == ["get_case", "resolve_case", "search_kb"]
