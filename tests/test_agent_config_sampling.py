"""``AgentConfig.temperature`` / ``max_tokens`` / ``tool_choice`` reach the model (1.90.0, F6).

All three were declared on ``AgentConfig``, documented in
``docs/agents/tools.md``, carried by the canonical agent fixture the plane
distributes — and read by nothing. ``temperature`` and ``max_tokens`` were never
looked at; ``tool_choice`` was handed to the tool loop, which never passed it on,
and ``LLMClient`` had no way to send it.

Since 1.90.0:

* ``temperature`` / ``max_tokens`` set on the agent go on every model call of
  the run (tool loop, structured re-ask, guardrail re-ask), over the
  ``LLMClient``'s own values; an explicit ``arun(..., temperature=)`` still wins.
* ``tool_choice`` is sent only when it is not ``"auto"`` — every request body an
  agent on the default sends is byte-for-byte what it was. ``"required"`` forces
  the **first** turn only (a model that must always call a tool never answers);
  ``"none"`` applies to every turn.
* Nothing is dropped for a model's sampling rules (1.89.0 decision D4): the
  provider stays the authority.

Real ``LLMClient`` → real ``httpx`` → a real localhost socket
(``tests.conftest.CaptureServer``) answering in each provider's wire format.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

from fastaiagent import Agent, AgentConfig, FunctionTool
from fastaiagent.llm.client import LLMClient


def _lookup(order: str) -> str:
    """Look an order up."""
    return f"order {order} shipped"


LOOKUP = FunctionTool(name="lookup", fn=_lookup)

# --- each provider's two answers: "call lookup" and "here is the answer" -----

_OPENAI_CALL = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"order": "7"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}
_OPENAI_TEXT = {
    "choices": [{"message": {"role": "assistant", "content": "shipped"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}
_ANTHROPIC_CALL = {
    "content": [{"type": "tool_use", "id": "t1", "name": "lookup", "input": {"order": "7"}}],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 5, "output_tokens": 3},
}
_ANTHROPIC_TEXT = {
    "content": [{"type": "text", "text": "shipped"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 3},
}
_GEMINI_CALL = {
    "candidates": [
        {
            "content": {"parts": [{"functionCall": {"name": "lookup", "args": {"order": "7"}}}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
}
_GEMINI_TEXT = {
    "candidates": [{"content": {"parts": [{"text": "shipped"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
}
_OLLAMA_TEXT = {
    "message": {"role": "assistant", "content": "shipped"},
    "done": True,
    "prompt_eval_count": 5,
    "eval_count": 3,
}

ANSWERS = {
    "openai": (_OPENAI_CALL, _OPENAI_TEXT),
    "anthropic": (_ANTHROPIC_CALL, _ANTHROPIC_TEXT),
    "gemini": (_GEMINI_CALL, _GEMINI_TEXT),
}
MODELS = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5", "gemini": "gemini-2.5-flash"}


def _client(provider: str, url: str, **kw: Any) -> LLMClient:
    return LLMClient(
        provider=provider, model=MODELS.get(provider, "llama3"), api_key="k", base_url=url, **kw
    )


def _bodies(capture: Any) -> list[dict[str, Any]]:
    return [r["body"] for r in capture.requests if isinstance(r["body"], dict)]


def _temperature(provider: str, body: dict[str, Any]) -> Any:
    if provider == "gemini":
        return (body.get("generationConfig") or {}).get("temperature")
    if provider == "ollama":
        return (body.get("options") or {}).get("temperature")
    return body.get("temperature")


def _max_tokens(provider: str, body: dict[str, Any]) -> Any:
    if provider == "gemini":
        return (body.get("generationConfig") or {}).get("maxOutputTokens")
    if provider == "ollama":
        return (body.get("options") or {}).get("num_predict")
    return body.get("max_completion_tokens", body.get("max_tokens"))


def _tool_choice(provider: str, body: dict[str, Any]) -> Any:
    if provider == "gemini":
        return ((body.get("toolConfig") or {}).get("functionCallingConfig") or {}).get("mode")
    return body.get("tool_choice")


# ---------------------------------------------------------------------------
# temperature / max_tokens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
def test_agent_temperature_and_max_tokens_reach_every_call(provider, capture_server) -> None:
    capture_server.set_response_sequence([(200, a) for a in ANSWERS[provider]])
    agent = Agent(
        name="cfg",
        llm=_client(provider, capture_server.url),
        tools=[LOOKUP],
        config=AgentConfig(temperature=0.2, max_tokens=64),
    )

    assert agent.run("where is 7?", trace=False).output == "shipped"

    bodies = _bodies(capture_server)
    assert len(bodies) == 2
    for body in bodies:
        assert _temperature(provider, body) == 0.2, body
        assert _max_tokens(provider, body) == 64, body


def test_ollama_receives_the_agent_temperature(capture_server) -> None:
    capture_server.set_response_sequence([(200, _OLLAMA_TEXT)])
    agent = Agent(
        name="cfg",
        llm=_client("ollama", capture_server.url),
        config=AgentConfig(temperature=0.3, max_tokens=32),
    )
    agent.run("hi", trace=False)
    body = _bodies(capture_server)[0]
    assert _temperature("ollama", body) == 0.3
    assert _max_tokens("ollama", body) == 32


def test_the_agent_value_beats_the_client_and_an_explicit_call_beats_both(capture_server) -> None:
    capture_server.set_response_sequence([(200, _OPENAI_TEXT), (200, _OPENAI_TEXT)])
    agent = Agent(
        name="cfg",
        llm=_client("openai", capture_server.url, temperature=0.9, max_tokens=500),
        config=AgentConfig(temperature=0.2, max_tokens=64),
    )
    agent.run("hi", trace=False)
    agent.run("hi", trace=False, temperature=0.5)

    first, second = _bodies(capture_server)
    assert (first["temperature"], first["max_completion_tokens"]) == (0.2, 64)
    assert second["temperature"] == 0.5


def test_unset_fields_leave_the_request_as_it_was(capture_server) -> None:
    capture_server.set_response_sequence([(200, _OPENAI_TEXT)])
    agent = Agent(name="cfg", llm=_client("openai", capture_server.url, temperature=0.9))
    agent.run("hi", trace=False)
    body = _bodies(capture_server)[0]
    assert body["temperature"] == 0.9, "the client's own value must still apply"
    assert "max_completion_tokens" not in body and "tool_choice" not in body


def test_the_structured_reask_carries_the_agent_temperature(capture_server) -> None:
    from pydantic import BaseModel

    class Out(BaseModel):
        answer: str

    bad = {
        "choices": [
            {"message": {"role": "assistant", "content": "not json"}, "finish_reason": "stop"}
        ],
        "usage": {},
    }
    good = {
        "choices": [
            {
                "message": {"role": "assistant", "content": '{"answer": "x"}'},
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }
    capture_server.set_response_sequence([(200, bad), (200, good)])
    agent = Agent(
        name="cfg",
        llm=_client("openai", capture_server.url),
        output_type=Out,
        config=AgentConfig(temperature=0.1),
    )
    assert agent.run("hi", trace=False).parsed == Out(answer="x")
    assert [b["temperature"] for b in _bodies(capture_server)] == [0.1, 0.1]


def test_astream_carries_the_agent_temperature(capture_server) -> None:
    sse = (
        'data: {"choices":[{"delta":{"content":"shipped"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )
    capture_server.set_response_sequence([(200, sse, "text/event-stream")])
    agent = Agent(
        name="cfg",
        llm=_client("openai", capture_server.url),
        config=AgentConfig(temperature=0.2, max_tokens=64, tool_choice="none"),
        tools=[LOOKUP],
    )

    async def drain() -> None:
        async for _ in agent.astream("hi", trace=False):
            pass

    asyncio.run(drain())
    body = _bodies(capture_server)[0]
    assert (body["temperature"], body["max_completion_tokens"]) == (0.2, 64)
    assert body["tool_choice"] == "none"


# ---------------------------------------------------------------------------
# tool_choice
# ---------------------------------------------------------------------------

EXPECTED_CHOICE = {
    "openai": {"required": "required", "none": "none"},
    "anthropic": {"required": {"type": "any"}, "none": {"type": "none"}},
    "gemini": {"required": "ANY", "none": "NONE"},
}


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
def test_auto_sends_nothing(provider, capture_server) -> None:
    capture_server.set_response_sequence([(200, a) for a in ANSWERS[provider]])
    agent = Agent(name="cfg", llm=_client(provider, capture_server.url), tools=[LOOKUP])
    agent.run("where is 7?", trace=False)
    for body in _bodies(capture_server):
        assert _tool_choice(provider, body) is None, body


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
def test_required_forces_the_first_turn_only(provider, capture_server) -> None:
    capture_server.set_response_sequence([(200, a) for a in ANSWERS[provider]])
    agent = Agent(
        name="cfg",
        llm=_client(provider, capture_server.url),
        tools=[LOOKUP],
        config=AgentConfig(tool_choice="required"),
    )
    assert agent.run("where is 7?", trace=False).output == "shipped"

    first, second = _bodies(capture_server)
    assert _tool_choice(provider, first) == EXPECTED_CHOICE[provider]["required"]
    assert _tool_choice(provider, second) is None, "a forced second turn could never answer"


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
def test_none_applies_to_every_turn(provider, capture_server) -> None:
    capture_server.set_response_sequence([(200, ANSWERS[provider][1])])
    agent = Agent(
        name="cfg",
        llm=_client(provider, capture_server.url),
        tools=[LOOKUP],
        config=AgentConfig(tool_choice="none"),
    )
    agent.run("where is 7?", trace=False)
    assert _tool_choice(provider, _bodies(capture_server)[0]) == EXPECTED_CHOICE[provider]["none"]


def test_a_provider_without_tool_choice_warns_and_sends_nothing(capture_server, caplog) -> None:
    capture_server.set_response_sequence([(200, _OLLAMA_TEXT)])
    agent = Agent(
        name="cfg",
        llm=_client("ollama", capture_server.url),
        tools=[LOOKUP],
        config=AgentConfig(tool_choice="required"),
    )
    with caplog.at_level(logging.WARNING, logger="fastaiagent"):
        agent.run("hi", trace=False)
    assert "tool_choice" not in json.dumps(_bodies(capture_server)[0])
    assert any("tool_choice" in r.getMessage() for r in caplog.records)


def test_an_unknown_tool_choice_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="tool_choice"):
        AgentConfig(tool_choice="sometimes")


def test_the_canonical_fixture_still_loads() -> None:
    """The plane distributes this shape; its values are now honoured, not dropped."""
    from pathlib import Path

    data = json.loads(
        (Path(__file__).parent.parent / "fixtures" / "agent_with_guardrails.json").read_text()
    )
    agent = Agent.from_dict(data)
    assert (agent.config.temperature, agent.config.max_tokens) == (0.7, 1000)
    assert agent.config.tool_choice == "auto"
