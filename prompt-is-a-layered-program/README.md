# A Prompt Is a Layered Program

Companion code for "A Prompt Is a Layered Program: Mapping Prompt Engineering Techniques to the Layer They Change" (<https://tensorup.ai/posts/prompt-is-a-layered-program>). It keeps a prompt as six provider-neutral layers in a `PromptProgram`, compiles them per model with `compile_request()` and a `ModelProfile`, and sends them through LiteLLM's `completion()` from `run()`.

## What it shows

`PromptProgram` owns the layers (instructions, tools, examples, context, task, output contract); `ModelProfile` owns every provider difference; `compile_request()` places each layer in its request slot or raises `UnsupportedLayerError`; `run()` validates the output with the full Pydantic model and retries within an attempt budget. The tests reproduce the post's failure modes: unsupported parameters raising instead of being dropped, schema constraints stripped from the wire but still enforced on the client, and a bounded retry ending in `ContractError`.

This sample uses LiteLLM as its only provider layer, because the post is about it. It does not follow the standard `OpenAIModel` / `AnthropicModel` / `TENSORUP_PROVIDER` layout of other TensorUp samples; the provider is chosen by the LiteLLM model id.

## Run the tests (no API key needed)

Requires [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`). It installs Python 3.11+ itself if needed.

```bash
git clone https://github.com/tensorupai/tensorup-examples
cd tensorup-examples/prompt-is-a-layered-program
uv sync
uv run pytest -q
```

The tests use `FakeModel`, a scripted stand-in for the model, and LiteLLM's bundled model map, so they run offline and never call a provider.

## Run it live

Set one key in your shell (the names are in `.env.example`; the sample does not load `.env` itself):

| Provider | Variable | Where to get it |
| --- | --- | --- |
| OpenAI (default) | `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| Anthropic | `ANTHROPIC_API_KEY` | https://console.anthropic.com → API Keys |

Then:

```bash
TENSORUP_LIVE=1 uv run python -m promptlayers.triage
TENSORUP_LIVE=1 TENSORUP_MODEL=anthropic/<model-id> uv run python -m promptlayers.triage
```

`TENSORUP_LIVE=1` is required for any real call; without it the program exits with a message instead of spending money. `TENSORUP_MODEL` takes any LiteLLM model id; the default is `openai/gpt-4o-2024-08-06`.

## Layout

```
src/promptlayers/
  layers.py     PromptProgram, Example (the six layers)
  profiles.py   ModelProfile, profile_from_litellm()
  compile.py    compile_request(), output_schema(), UnsupportedLayerError
  run.py        Model protocol, LiteLLMModel, run(), ContractError, ToolCallError
  triage.py     example program and live entry point
tests/
  conftest.py   FakeModel
```

Dependencies are pinned in `pyproject.toml` and `uv.lock`; `uv sync` reproduces the exact environment the post was tested with.
