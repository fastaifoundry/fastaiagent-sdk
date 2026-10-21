"""How structured output is asked for, and streamed (1.91.0, 2G + X21).

``AgentConfig(output_mode=...)`` for an agent with ``output_type``:

* ``"auto"`` (default) — what each provider did before: OpenAI/Azure native
  ``response_format``, Anthropic the schema in the prompt, Gemini its native
  schema (prompted on tool turns, X20), Ollama ``format`` — and Bedrock, which
  ignored structured output entirely (X21), now the schema in the prompt.
* ``"native"`` — the provider's own schema-constrained output, including
  Anthropic ``output_config.format`` and Bedrock ``outputConfig.textFormat``.
* ``"tool"`` — a ``final_result`` tool carrying the schema; calling it ends the
  run with ``parsed`` = its arguments. Offered, not forced.
* ``"prompt"`` — the schema in the prompt only, on every provider.

While a structured run streams, ``PartialOutput(data=...)`` carries the reply so
far, parsed as incomplete JSON.

Real ``Agent`` → real ``LLMClient`` (real ``boto3`` for Bedrock) →
``tests.conftest.CaptureServer``; ``FunctionModel`` where only the loop matters.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import BaseModel, Field

from fastaiagent import Agent, AgentConfig, FunctionTool, LLMClient
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import ToolCall
from fastaiagent.llm.stream import PartialOutput
from fastaiagent.testing import FunctionModel


class Forecast(BaseModel):
    city: str = Field(max_length=40)
    celsius: float = Field(ge=-90, le=60)


def get_weather(city: str) -> str:
    """Current weather for a city."""
    return "24C, sunny"


ANTHROPIC_JSON = {
    "id": "m1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": '{"city": "Rome", "celsius": 24}'}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 9},
}
OPENAI_JSON = {
    "id": "c1",
    "model": "gpt-5-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": '{"city": "Rome", "celsius": 24}'},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14},
}


def _agent(llm: Any, mode: str | None = None, **kw: Any) -> Agent:
    config = AgentConfig(output_mode=mode) if mode else AgentConfig()
    return Agent(name="forecaster", llm=llm, output_type=Forecast, config=config, **kw)


def _body(capture: Any, index: int = 0) -> dict[str, Any]:
    return capture.requests[index]["body"]


def _system_text(body: dict[str, Any]) -> str:
    system = body.get("system")
    if isinstance(system, list):
        return " ".join(b.get("text", "") for b in system if isinstance(b, dict))
    if system:
        return str(system)
    for m in body.get("messages", []):
        if m.get("role") == "system":
            return str(m.get("content"))
    return ""


# ── the vocabulary ─────────────────────────────────────────────────────────


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError):
        AgentConfig(output_mode="json")


def test_output_mode_is_in_to_dict_only_when_not_auto() -> None:
    llm = LLMClient(provider="openai", model="gpt-5-mini")
    assert "output_mode" not in _agent(llm).to_dict()["config"]
    assert _agent(llm, "tool").to_dict()["config"]["output_mode"] == "tool"


# ── auto: unchanged, except X21 ────────────────────────────────────────────


def test_auto_on_openai_is_the_native_response_format(capture_server) -> None:
    capture_server.set_response_sequence([(200, OPENAI_JSON)])
    llm = LLMClient(provider="openai", model="gpt-5-mini", api_key="k", base_url=capture_server.url)
    result = _agent(llm).run("Rome", trace=False)
    assert _body(capture_server)["response_format"]["type"] == "json_schema"
    assert result.parsed == Forecast(city="Rome", celsius=24)


def test_auto_on_anthropic_is_the_prompt(capture_server) -> None:
    capture_server.set_response_sequence([(200, ANTHROPIC_JSON)])
    llm = LLMClient(
        provider="anthropic", model="claude-haiku-5-5", api_key="k", base_url=capture_server.url
    )
    _agent(llm).run("Rome", trace=False)
    body = _body(capture_server)
    assert "output_config" not in body
    assert "matching this schema" in _system_text(body)


def test_auto_on_bedrock_now_asks_for_the_schema(capture_server, monkeypatch) -> None:
    """X21: ``_call_bedrock`` never read ``response_format``."""
    pytest.importorskip("boto3")
    _dummy_aws(monkeypatch)
    capture_server.set_response_sequence([(200, _bedrock('{"city": "Rome", "celsius": 24}'))])
    result = _agent(_bedrock_client(capture_server.url)).run("Rome", trace=False)
    system = " ".join(b.get("text", "") for b in _body(capture_server)["system"])
    assert "matching this schema" in system and "celsius" in system
    assert result.parsed == Forecast(city="Rome", celsius=24)


# ── native ─────────────────────────────────────────────────────────────────


def test_native_on_anthropic_is_output_config_format(capture_server) -> None:
    capture_server.set_response_sequence([(200, ANTHROPIC_JSON)])
    llm = LLMClient(
        provider="anthropic",
        model="claude-haiku-5-5",
        api_key="k",
        base_url=capture_server.url,
        reasoning="low",
    )
    result = _agent(llm, "native").run("Rome", trace=False)
    body = _body(capture_server)
    fmt = body["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False
    # Constraints Anthropic refuses move into the description; pydantic still checks.
    props = fmt["schema"]["properties"]
    assert "maxLength" not in props["city"] and "40" in props["city"]["description"]
    assert "minimum" not in props["celsius"] and "maximum" not in props["celsius"]
    assert body["output_config"]["effort"] == "low", "reasoning's effort kept beside it"
    assert "matching this schema" not in _system_text(body)
    assert result.parsed == Forecast(city="Rome", celsius=24)


def test_native_on_bedrock_is_output_config_text_format(capture_server, monkeypatch) -> None:
    pytest.importorskip("boto3")
    _dummy_aws(monkeypatch)
    capture_server.set_response_sequence([(200, _bedrock('{"city": "Rome", "celsius": 24}'))])
    result = _agent(_bedrock_client(capture_server.url), "native").run("Rome", trace=False)
    text_format = _body(capture_server)["outputConfig"]["textFormat"]
    assert text_format["type"] == "json_schema"
    schema = json.loads(text_format["structure"]["jsonSchema"]["schema"])
    assert schema["properties"]["celsius"]["type"] == "number"
    assert result.parsed == Forecast(city="Rome", celsius=24)


# ── prompt ─────────────────────────────────────────────────────────────────


def test_prompt_mode_sends_no_provider_parameter(capture_server) -> None:
    capture_server.set_response_sequence([(200, OPENAI_JSON)])
    llm = LLMClient(provider="openai", model="gpt-5-mini", api_key="k", base_url=capture_server.url)
    result = _agent(llm, "prompt").run("Rome", trace=False)
    body = _body(capture_server)
    assert "response_format" not in body
    assert "matching this schema" in _system_text(body)
    assert result.parsed == Forecast(city="Rome", celsius=24)


# ── tool ───────────────────────────────────────────────────────────────────


def _tool_mode_model(*, text_answer: bool = False) -> FunctionModel:
    turns = {"n": 0}

    def respond(messages):
        turns["n"] += 1
        if turns["n"] == 1:
            return "", [{"name": "get_weather", "arguments": {"city": "Rome"}}]
        if text_answer:
            return '{"city": "Rome", "celsius": 24}'
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="f1", name="final_result", arguments={"city": "Rome", "celsius": 24}),
                ToolCall(id="g2", name="get_weather", arguments={"city": "Oslo"}),
            ],
        )

    model = FunctionModel(respond)
    model.turns = turns  # type: ignore[attr-defined]
    return model


def test_tool_mode_offers_final_result_and_ends_on_it() -> None:
    ran: list[str] = []

    def weather(city: str) -> str:
        """Current weather for a city."""
        ran.append(city)
        return "24C"

    model = _tool_mode_model()
    agent = _agent(model, "tool", tools=[FunctionTool(name="get_weather", fn=weather)])
    result = agent.run("Rome")
    offered = [t["function"]["name"] for t in model.calls[0]["tools"]]
    assert offered == ["get_weather", "final_result"]
    final = next(t for t in model.calls[0]["tools"] if t["function"]["name"] == "final_result")
    assert final["function"]["parameters"]["properties"]["celsius"]["type"] == "number"
    assert result.parsed == Forecast(city="Rome", celsius=24)
    assert json.loads(result.output) == {"city": "Rome", "celsius": 24}
    assert model.turns["n"] == 2, "the final_result call ended the run"
    assert ran == ["Rome"], "nothing after final_result ran"
    assert "response_format" not in model.calls[0]["kwargs"]


def test_tool_mode_still_takes_a_json_text_answer() -> None:
    model = _tool_mode_model(text_answer=True)
    agent = _agent(model, "tool", tools=[FunctionTool(name="get_weather", fn=get_weather)])
    assert agent.run("Rome").parsed == Forecast(city="Rome", celsius=24)


def test_tool_mode_streams_and_ends_on_final_result() -> None:
    model = _tool_mode_model()
    agent = _agent(model, "tool", tools=[FunctionTool(name="get_weather", fn=get_weather)])

    async def drain() -> list[Any]:
        return [e async for e in agent.astream("Rome")]

    events = asyncio.run(drain())
    partials = [e for e in events if isinstance(e, PartialOutput)]
    assert partials and partials[-1].data == {"city": "Rome", "celsius": 24}
    assert agent.stream("Rome").parsed == Forecast(city="Rome", celsius=24)


def test_a_user_tool_called_final_result_is_refused_in_tool_mode() -> None:
    with pytest.raises(ValueError, match="final_result"):
        _agent(
            FunctionModel(lambda m: "x"),
            "tool",
            tools=[FunctionTool(name="final_result", fn=get_weather)],
        )


# ── PartialOutput ──────────────────────────────────────────────────────────


def _sse_openai(*pieces: str) -> str:
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]} for p in pieces
    ]
    chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def test_partial_output_streams_the_reply_so_far(capture_server) -> None:
    capture_server.set_response_sequence(
        [(200, _sse_openai('{"city": "Ro', 'me", "cel', 'sius": 24}'), "text/event-stream")]
    )
    llm = LLMClient(provider="openai", model="gpt-5-mini", api_key="k", base_url=capture_server.url)

    async def drain() -> list[Any]:
        return [e async for e in _agent(llm).astream("Rome", trace=False)]

    data = [e.data for e in asyncio.run(drain()) if isinstance(e, PartialOutput)]
    assert data[0] == {"city": "Ro"}
    assert data[-1] == {"city": "Rome", "celsius": 24}
    assert all(a != b for a, b in zip(data, data[1:])), "an unchanged value is not repeated"


def test_text_before_a_tool_call_is_not_partial_output(capture_server) -> None:
    def respond(messages):
        if not any(m.role.value == "tool" for m in messages):
            return "Let me check.", [{"name": "get_weather", "arguments": {"city": "Rome"}}]
        return '{"city": "Rome", "celsius": 24}'

    agent = _agent(FunctionModel(respond), tools=[FunctionTool(name="get_weather", fn=get_weather)])

    async def drain() -> list[Any]:
        return [e async for e in agent.astream("Rome")]

    data = [e.data for e in asyncio.run(drain()) if isinstance(e, PartialOutput)]
    assert data == [{"city": "Rome", "celsius": 24}]


def test_an_agent_without_output_type_streams_no_partials() -> None:
    agent = Agent(name="plain", llm=FunctionModel(lambda m: '{"a": 1}'))

    async def drain() -> list[Any]:
        return [e async for e in agent.astream("x")]

    assert not [e for e in asyncio.run(drain()) if isinstance(e, PartialOutput)]


# ── Bedrock helpers (real boto3, dummy credentials) ────────────────────────


def _dummy_aws(monkeypatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDUMMY")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "dummy-secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def _bedrock(text: str) -> dict[str, Any]:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 7, "outputTokens": 3, "totalTokens": 10},
        "metrics": {"latencyMs": 1},
    }


def _bedrock_client(url: str) -> LLMClient:
    return LLMClient(provider="bedrock", model="us.anthropic.claude-haiku-5-5-v1:0", base_url=url)
