"""Flexible structured output: non-model types, strict mode, retry-on-failure.

Runs without API keys. A scripted local LLMClient (returns pre-baked replies —
not a mock of our own code) drives the real Agent parse/retry path.
"""

from __future__ import annotations

import enum
import json

import pytest
from pydantic import BaseModel

from fastaiagent import Agent, AgentConfig
from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.llm.structured import OutputSpec


class Person(BaseModel):
    name: str
    age: int


class Priority(str, enum.Enum):
    low = "low"
    high = "high"


class Addr(BaseModel):
    street: str
    zip: str | None = None


class Customer(BaseModel):
    name: str
    addr: Addr


def _scripted(replies: list[str], provider: str = "mock") -> LLMClient:
    class _LLM(LLMClient):
        def __init__(self) -> None:
            super().__init__(provider=provider, model="mock")
            self._i = 0

        async def acomplete(self, messages, tools=None, **kwargs) -> LLMResponse:
            reply = replies[min(self._i, len(replies) - 1)]
            self._i += 1
            return LLMResponse(content=reply, finish_reason="stop", usage={"total_tokens": 5})

    return _LLM()


# ─── OutputSpec: schema shaping ─────────────────────────────────────────────


def test_model_schema_is_unchanged() -> None:
    spec = OutputSpec(Person)
    assert spec.wrapped is False
    rf = spec.response_format(strict=False)
    assert rf["json_schema"]["schema"] == Person.model_json_schema()  # byte-identical
    assert rf["json_schema"]["name"] == "Person"


def test_list_output_is_wrapped_and_unwrapped() -> None:
    spec = OutputSpec(list[Person])
    assert spec.wrapped is True
    schema = spec.response_format(strict=False)["json_schema"]["schema"]
    assert schema["type"] == "object"
    assert list(schema["properties"]) == ["value"]
    # Unwraps the {"value": [...]} envelope...
    v, err = spec.parse(json.dumps({"value": [{"name": "A", "age": 1}]}))
    assert err is None and v == [Person(name="A", age=1)]
    # ...and also tolerates a bare array (fallback providers).
    v2, _ = spec.parse(json.dumps([{"name": "B", "age": 2}]))
    assert v2 == [Person(name="B", age=2)]


def test_primitive_output() -> None:
    spec = OutputSpec(int)
    assert spec.wrapped is True
    v, err = spec.parse(json.dumps({"value": 42}))
    assert v == 42 and err is None


def test_parse_errors_are_reported() -> None:
    spec = OutputSpec(Person)
    v, err = spec.parse("not json")
    assert v is None and "not valid JSON" in err
    v2, err2 = spec.parse(json.dumps({"name": "x"}))  # missing required 'age'
    assert v2 is None and "did not match" in err2


def test_strict_transform() -> None:
    schema = OutputSpec(Customer).response_format(strict=True)["json_schema"]
    assert schema["strict"] is True
    s = schema["schema"]
    assert s["additionalProperties"] is False
    assert set(s["required"]) == {"name", "addr"}
    addr = s["$defs"]["Addr"]
    assert addr["additionalProperties"] is False
    # Optional 'zip' is forced required (nullable) under strict.
    assert set(addr["required"]) == {"street", "zip"}


# ─── Agent integration ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_parses_list_output() -> None:
    agent = Agent(
        name="lister",
        llm=_scripted([json.dumps({"value": [{"name": "A", "age": 1}, {"name": "B", "age": 2}]})]),
        output_type=list[Person],
    )
    result = await agent.arun("x", trace=False)
    assert result.parsed == [Person(name="A", age=1), Person(name="B", age=2)]


@pytest.mark.asyncio
async def test_agent_parses_enum() -> None:
    agent = Agent(name="e", llm=_scripted([json.dumps({"value": "high"})]), output_type=Priority)
    result = await agent.arun("x", trace=False)
    assert result.parsed is Priority.high


@pytest.mark.asyncio
async def test_retry_recovers_and_counts_tokens() -> None:
    agent = Agent(
        name="retry",
        llm=_scripted(["this is not json", json.dumps({"name": "Bob", "age": 9})]),
        output_type=Person,
        config=AgentConfig(output_retries=2),
    )
    result = await agent.arun("x", trace=False)
    assert result.parsed == Person(name="Bob", age=9)
    assert result.tokens_used == 10  # first attempt + one retry, 5 tokens each


@pytest.mark.asyncio
async def test_retries_disabled_preserves_old_behavior() -> None:
    agent = Agent(
        name="noretry",
        llm=_scripted(["not json", json.dumps({"name": "Z", "age": 1})]),
        output_type=Person,
        config=AgentConfig(output_retries=0),
    )
    result = await agent.arun("x", trace=False)
    assert result.parsed is None  # unchanged legacy behavior
    assert result.output == "not json"


@pytest.mark.asyncio
async def test_retry_exhausted_yields_none() -> None:
    agent = Agent(
        name="bad",
        llm=_scripted(["nope"]),  # always bad
        output_type=Person,
        config=AgentConfig(output_retries=2),
    )
    result = await agent.arun("x", trace=False)
    assert result.parsed is None


@pytest.mark.asyncio
async def test_llmclient_output_type_populates_parsed() -> None:
    # A double that overrides the raw call so the acomplete() wrapper (which
    # builds response_format and parses) runs end to end.
    class _Client(LLMClient):
        def __init__(self, reply: str) -> None:
            super().__init__(provider="mock", model="mock")
            self._reply = reply

        async def _acomplete_raw(self, messages, tools=None, **kwargs) -> LLMResponse:
            assert "response_format" in kwargs  # wrapper injected the schema
            return LLMResponse(content=self._reply, finish_reason="stop")

    client = _Client(json.dumps({"name": "Ada", "age": 36}))
    resp = await client.acomplete([], output_type=Person)
    assert resp.parsed == Person(name="Ada", age=36)

    list_client = _Client(json.dumps({"value": [{"name": "X", "age": 1}]}))
    resp2 = await list_client.acomplete([], output_type=list[Person])
    assert resp2.parsed == [Person(name="X", age=1)]

    # Without output_type, parsed stays None and no response_format is required.
    class _Plain(LLMClient):
        def __init__(self) -> None:
            super().__init__(provider="mock", model="mock")

        async def _acomplete_raw(self, messages, tools=None, **kwargs) -> LLMResponse:
            return LLMResponse(content="hi", finish_reason="stop")

    resp3 = await _Plain().acomplete([])
    assert resp3.parsed is None


def test_strict_gated_to_openai_provider() -> None:
    mock_agent = Agent(
        name="m", llm=_scripted(["{}"], provider="mock"), output_type=Person,
        config=AgentConfig(strict_output=True),
    )
    assert "strict" not in mock_agent._build_response_format()["json_schema"]

    oai_agent = Agent(
        name="o", llm=_scripted(["{}"], provider="openai"), output_type=Person,
        config=AgentConfig(strict_output=True),
    )
    assert oai_agent._build_response_format()["json_schema"]["strict"] is True
