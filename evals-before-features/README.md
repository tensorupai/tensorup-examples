# Evals Before Features: How to Test a Strands Agent You Cannot Unit-Test

Companion code for "Evals Before Features: How to Test a Strands Agent You Cannot Unit-Test" (<https://tensorup.ai/posts/evals-before-features>). It implements eval-first development for a support chatbot: the eval cases in `cases.py` define the business behaviour, a Strands harness agent (`create_harness`) answers from a mocked knowledge base and resolves support cases, and Strands Evals scores every run with deterministic checks and LLM judges before a release gate passes or fails.

## What it shows

`cases.py` is the spec, written before the agent's instructions. `evals.py` runs each case through the agent, captures its trace as a Strands Evals `Session`, and scores it with two deterministic evaluators (`ToolPolicy`, `CitesSource`) on every commit and four judge evaluators (correctness, relevance, helpfulness, faithfulness) before a release. The tests replay two scripted models: a baseline that passes the gate, and an "upgraded" model that answers without searching the knowledge base and closes a case the customer did not confirm, which the gate rejects. `support.py` enforces case ownership and idempotent resolution in code, so evals check behaviour and code checks limits.

## Run the tests (no API key needed)

Requires [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`). It installs Python 3.11+ itself if needed.

```bash
git clone https://github.com/tensorupai/tensorup-examples
cd tensorup-examples/evals-before-features
uv sync
uv run pytest -q
```

The tests use `FakeModel`, a scripted stand-in for the model, so they run offline and never call a provider.

## Run it live

Copy `.env.example` to `.env` and fill in one key:

| Provider | Variable | Where to get it |
| --- | --- | --- |
| OpenAI (default) | `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| Anthropic | `ANTHROPIC_API_KEY` | https://console.anthropic.com → API Keys |

Then:

```bash
# Run the full eval suite (deterministic checks and LLM judges); exits 1 if the gate fails
TENSORUP_LIVE=1 uv run python -m supportevals eval
# Ask the support agent one question
TENSORUP_LIVE=1 uv run python -m supportevals chat "Can I still return headphones I bought 20 days ago?"
TENSORUP_LIVE=1 TENSORUP_PROVIDER=anthropic uv run python -m supportevals eval
```

A live `eval` makes about 21 agent runs and 84 judge ratings one after another, and prints a line as each case-trial finishes. strands-agents 1.64.0 logs an `aclose(): asynchronous generator is already running` error after each OpenAI call without affecting the result; `__main__.py` filters that one message.

`TENSORUP_LIVE=1` is required for any real call; without it the program exits with a message instead of spending money. `TENSORUP_MODEL` overrides the agent's model (`DEFAULT_OPENAI_MODEL` / `DEFAULT_ANTHROPIC_MODEL`) and `TENSORUP_JUDGE_MODEL` the judge's (`DEFAULT_OPENAI_JUDGE` / `DEFAULT_ANTHROPIC_JUDGE`). The judge uses the same provider and key as the agent.

## Layout

```
src/supportevals/
  cases.py      the eval cases: the business behaviour, written first
  support.py    SupportDesk, the mocked knowledge base and the three tools
  agent.py      the support agent, built with create_harness
  evals.py      task function, evaluators, thresholds and the release gate
  model.py      FakeModel and the OpenAI and Anthropic models for agent and judge
  __main__.py   live entry point (eval, chat)
tests/
```

Dependencies are pinned in `pyproject.toml` and `uv.lock`; `uv sync` reproduces the exact environment the post was tested with.
