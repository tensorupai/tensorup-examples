import pytest

from loopstop import __main__ as cli
from loopstop.environment import BAD_DB_URL, GOOD_DB_URL, ServiceEnv
from loopstop.model import MissingKeyError, build_model
from loopstop.review import GoalCheck


def state(db_url: str, replicas: int) -> dict:
    return {"config": {"db_url": db_url}, "replicas": replicas, "healthy": db_url == GOOD_DB_URL}


def test_goal_check_outcomes():
    check = GoalCheck(target_replicas=3)
    bad = state(BAD_DB_URL, 1)
    assert check.review(bad, state(GOOD_DB_URL, 3)).outcome == "pass"
    assert check.review(bad, state(GOOD_DB_URL, 1)).reasons == ("replicas=1, want 3",)
    assert check.review(bad, bad).outcome == "stalled"


def test_restart_changes_no_goal_state():
    env = ServiceEnv()
    before = env.fingerprint()
    env.restarts += 1
    assert env.fingerprint() == before


def test_cli_refuses_without_live_flag(monkeypatch, capsys):
    monkeypatch.delenv("TENSORUP_LIVE", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    assert cli.main() == 2
    assert "TENSORUP_LIVE=1" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("provider", "var"), [("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")]
)
def test_missing_key_names_the_variable(monkeypatch, provider, var):
    monkeypatch.setenv("TENSORUP_PROVIDER", provider)
    monkeypatch.delenv(var, raising=False)
    with pytest.raises(MissingKeyError, match=var):
        build_model()


def test_build_model_uses_default_ids(monkeypatch):
    monkeypatch.delenv("TENSORUP_MODEL", raising=False)
    monkeypatch.setenv("TENSORUP_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert build_model().get_config()["model_id"] == "claude-haiku-5-5"
    monkeypatch.setenv("TENSORUP_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert build_model().get_config()["model_id"] == "gpt-6-luna"
