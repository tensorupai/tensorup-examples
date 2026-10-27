# Why Your Agent Loops Forever: Put the Review Step Outside the Executor

Companion code for "Why Your Agent Loops Forever: Put the Review Step Outside the Executor" (<https://tensorup.ai/posts/why-your-agent-loops-forever-put-the-review-step-outside-the-executor/>). It implements a planner/executor split on Strands Agents 1.62.0 in which a deterministic `GoalCheck` reviews system state after each executor run and reports only to the planner, while `NoProgressGuard` and per-call `Limits` bound the executor's own loop.

## What it shows

`ServiceEnv` is a service that is down because of a bad `db_url`; `restart_service` always returns "restart ok" and changes nothing the goal can see. `Orchestrator` runs planner, executor and `GoalCheck` in rounds and stops on pass, on a round with no state change (`stalled`), or when its round budget runs out. `NoProgressGuard` cancels a tool call that repeats from an unchanged state and ends the executor run. The tests reproduce the restart loop, a transcript that claims success the state does not show, the turn limit, the round budget, and check that the reviewer's findings reach the planner and never the executor.

## Run the tests (no API key needed)

Requires [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`). It installs Python 3.11+ itself if needed.

```bash
git clone https://github.com/tensorupai/tensorup-examples
cd tensorup-examples/why-agents-loop-forever
uv sync
uv run pytest -q
```

The tests use `FakeModel`, a scripted Strands `Model`, so they run offline and never call a provider.

## Run it live

Copy `.env.example` to `.env` and fill in one key:

| Provider | Variable | Where to get it |
| --- | --- | --- |
| OpenAI (default) | `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| Anthropic | `ANTHROPIC_API_KEY` | https://console.anthropic.com → API Keys |

Then:

```bash
TENSORUP_LIVE=1 uv run python -m loopstop "Get the orders service healthy and running 3 replicas."
TENSORUP_LIVE=1 TENSORUP_PROVIDER=anthropic uv run python -m loopstop
```

`TENSORUP_LIVE=1` is required for any real call; without it the program exits with a message instead of spending money. `TENSORUP_MODEL` overrides the default model (`DEFAULT_OPENAI_MODEL` / `DEFAULT_ANTHROPIC_MODEL`).

## Layout

```
src/loopstop/
  model.py         FakeModel, build_model() for Strands' OpenAIResponsesModel / AnthropicModel
  environment.py   ServiceEnv (snapshot, fingerprint) and the executor's tools
  guard.py         NoProgressGuard, a BeforeToolCallEvent hook
  review.py        GoalCheck and Verdict, the deterministic review step
  orchestrator.py  Orchestrator: planner -> executor -> GoalCheck -> planner
  __main__.py      live entry point
tests/
```

Dependencies are pinned in `pyproject.toml` and `uv.lock`; `uv sync` reproduces the exact environment the post was tested with.
