from dataclasses import replace
from typing import Literal

import pytest
from pydantic import BaseModel

from promptlayers.compile import UnsupportedLayerError, compile_request, output_schema
from promptlayers.profiles import NON_PORTABLE_KEYWORDS, ModelProfile
from promptlayers.triage import Triage, triage_program

OPENAI = ModelProfile(model="openai/test-model")
STRICT_PROVIDER = ModelProfile(
    model="anthropic/test-model", rejected_schema_keywords=NON_PORTABLE_KEYWORDS
)


def test_layers_land_in_their_request_slots() -> None:
    program = triage_program("Exports fail.", policy_docs=("P1 means outage.",))
    req = compile_request(program, OPENAI)
    roles = [m["role"] for m in req["messages"]]
    assert roles == ["system", "user", "assistant", "user"]  # L1, L3 pair, L4+L5
    last = req["messages"][-1]["content"]
    assert last.index("<document>") < last.index("<task>")  # context above the task
    assert req["response_format"]["json_schema"]["strict"] is True
    assert "tools" not in req


def test_tools_layer_is_passed_through() -> None:
    tool = {"type": "function", "function": {"name": "lookup_order", "parameters": {}}}
    program = replace(triage_program("Where is order 42?"), tools=(tool,))
    assert compile_request(program, OPENAI)["tools"] == [tool]


def test_schema_is_strict_and_closed() -> None:
    schema = output_schema(Triage, frozenset())
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["category", "priority", "summary"]
    assert schema["properties"]["summary"]["maxLength"] == 120


def test_rejected_keywords_are_stripped_from_the_wire_schema() -> None:
    schema = compile_request(triage_program("x"), STRICT_PROVIDER)["response_format"]
    props = schema["json_schema"]["schema"]["properties"]
    assert "maxLength" not in props["summary"]
    assert "minimum" not in props["priority"] and "maximum" not in props["priority"]


def test_field_named_like_a_keyword_survives_stripping() -> None:
    class Range(BaseModel):
        minimum: int
        maximum: int

    schema = output_schema(Range, NON_PORTABLE_KEYWORDS)
    assert set(schema["properties"]) == {"minimum", "maximum"}


def test_instruction_override_replaces_layer_one() -> None:
    profile = ModelProfile(model="openai/test-model", instruction_override="Short rules.")
    req = compile_request(triage_program("x"), profile)
    assert req["messages"][0]["content"] == "Short rules."


def test_temperature_on_a_model_without_it_raises_instead_of_dropping() -> None:
    reasoning = ModelProfile(model="openai/test-reasoning", supports_temperature=False)
    with pytest.raises(UnsupportedLayerError, match="temperature"):
        compile_request(triage_program("x"), reasoning, temperature=0.2)
    assert compile_request(triage_program("x"), OPENAI, temperature=0.2)["temperature"] == 0.2


def test_model_without_schema_output_raises() -> None:
    legacy = ModelProfile(model="openai/test-legacy", supports_json_schema=False)
    with pytest.raises(UnsupportedLayerError, match="JSON schema"):
        compile_request(triage_program("x"), legacy)


def test_open_ended_object_fields_are_refused() -> None:
    class Tags(BaseModel):
        labels: dict[str, str]

    with pytest.raises(UnsupportedLayerError, match="open-ended"):
        output_schema(Tags, frozenset())


def test_defaults_are_dropped_and_enum_values_kept_verbatim() -> None:
    class Ticket(BaseModel):
        status: Literal["open", "closed"] = "open"

    schema = output_schema(Ticket, NON_PORTABLE_KEYWORDS)
    assert "default" not in schema["properties"]["status"]
    assert schema["properties"]["status"]["enum"] == ["open", "closed"]
