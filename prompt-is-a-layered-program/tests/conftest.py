import os
from typing import Any

import pytest

# Use LiteLLM's bundled model map; never fetch it over the network in tests.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


class FakeModel:
    """Deterministic stand-in for a provider: returns scripted outputs, records requests."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[dict[str, Any]] = []

    def complete(self, request: dict[str, Any]) -> str:
        self.requests.append(request)
        return self.outputs.pop(0)


@pytest.fixture
def fake() -> type[FakeModel]:
    return FakeModel
