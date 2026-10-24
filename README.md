# tensorup-examples

Runnable code for posts on [TensorUp](https://tensorup.ai). Each folder is one post's sample: a self-contained Python project with its own pinned dependencies and offline tests.

| Sample | What it shows | Post |
| --- | --- | --- |
| [prompt-is-a-layered-program](prompt-is-a-layered-program/) | A prompt kept as six provider-neutral layers, compiled per model by `compile_request()` and a `ModelProfile`, sent through LiteLLM, with the output contract checked again by Pydantic | [A Prompt Is a Layered Program](https://tensorup.ai/posts/prompt-is-a-layered-program) |

## Run a sample

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.11+ itself if needed.

```bash
git clone https://github.com/tensorupai/tensorup-examples
cd tensorup-examples/<sample>
uv sync
uv run pytest -q
```

The tests run offline with a fake model and never call a provider. Each sample's README explains how to run it live; a real call always needs `TENSORUP_LIVE=1` and your own API key.

## License

MIT, see [LICENSE](LICENSE).
