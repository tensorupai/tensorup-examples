# How to Choose an Agent Control-Flow Pattern: Loop, Router, Graph, Planner/Executor, Supervisor

Companion code for "How to Choose an Agent Control-Flow Pattern: Loop, Router, Graph, Planner/Executor, Supervisor" (<https://tensorup.ai/posts/five-agent-control-flow-patterns>). It implements the five patterns on Strands Agents 1.63.0 over one fake web shop: a single `Agent` loop, a `Router` that validates a label before dispatching, a `GraphBuilder` checkout with deterministic `FunctionNode` steps, a `PlannerExecutor` with a fresh executor per step, and a supervisor whose workers are agents as tools.

## What it shows

`ShopSite` returns a full page snapshot (about 5,100 characters) from every `open_url` and `click`, as a browser tool does. The tests reproduce what each pattern does with that: the single loop resends every snapshot on every model call and, when the context overflows, trims the task first unless it is pinned; the planner never sees a snapshot and each executor starts empty; an unknown router label escalates instead of dispatching, while the substring edge conditions from the Strands docs fire both branches; the checkout graph places the order only after a deterministic cart check and stops its repair cycle at `max_node_executions`; an `as_tool()` worker never sees a budget the supervisor left out, and a typed handoff tool rejects a call without it.

## Run the tests (no API key needed)

Requires [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`). It installs Python 3.11+ itself if needed.

```bash
git clone https://github.com/tensorupai/tensorup-examples
cd tensorup-examples/five-agent-control-flow-patterns
uv sync
uv run pytest -q
```

The tests use `FakeModel`, a scripted Strands `Model`, so they run offline and never call a provider. `FakeModel(context_chars=...)` raises `ContextWindowOverflowException` like a provider with a full context window.

## Run it live

Copy `.env.example` to `.env` and fill in one key:

| Provider | Variable | Where to get it |
| --- | --- | --- |
| OpenAI (default) | `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| Anthropic | `ANTHROPIC_API_KEY` | https://console.anthropic.com → API Keys |

Then pick a pattern (`loop`, `router`, `graph`, `planner`, `supervisor`):

```bash
TENSORUP_LIVE=1 uv run python -m controlflow planner "Buy the studio headphones (sku hp-100) if they cost at most \$50."
TENSORUP_LIVE=1 TENSORUP_PROVIDER=anthropic uv run python -m controlflow graph
```

`TENSORUP_LIVE=1` is required for any real call; without it the program exits with a message instead of spending money. `TENSORUP_MODEL` overrides the default model (`DEFAULT_OPENAI_MODEL` / `DEFAULT_ANTHROPIC_MODEL`). OpenAI calls go through the Responses API (`OpenAIResponsesModel`).

## Layout

```
src/controlflow/
  model.py             FakeModel and the OpenAI and Anthropic providers behind Strands' Model
  shop.py              ShopSite and its browser-style tools (open_url, click)
  single_loop.py       pattern 1: one Agent, Limits, SlidingWindowConversationManager
  router.py            pattern 2: classify, validate the label, dispatch once
  graph_flow.py        pattern 3: GraphBuilder checkout with FunctionNode steps and a bounded cycle
  planner_executor.py  pattern 4: PlannerExecutor, a fresh executor Agent per step
  supervisor.py        pattern 5: as_tool() workers, and typed handoff tools that enforce max_price
  __main__.py          live entry point
tests/
```

Dependencies are pinned in `pyproject.toml` and `uv.lock`; `uv sync` reproduces the exact environment the post was tested with.
