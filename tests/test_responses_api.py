"""OpenAI's Responses API as an opt-in endpoint (1.91.0, 2C).

``LLMClient(provider="openai", api="responses")`` calls ``POST /responses``: its own
body builder and parser, returning the same ``LLMResponse`` — so the agent loop,
retries, spans, pricing and checkpoints are unchanged. Chat Completions stays the
default.

Why it exists, measured 2026-10-19: ``gpt-6-luna`` on Chat Completions answers a
request with tools and ``reasoning_effort`` other than ``"none"`` with HTTP 400 —
"Function tools with reasoning_effort are not supported for gpt-6-luna in
/v1/chat/completions. To use function tools, use /v1/responses …". The same request
works here, stateless (``store: false``), with the reasoning items passed back
across the tool loop as OpenAI requires.

Real ``LLMClient`` → real ``httpx`` → ``tests.conftest.CaptureServer`` answering in
the shapes the live API returned (probe outputs kept with the 1.91.0 work).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from fastaiagent import Agent, FunctionTool, SQLiteCheckpointer
from fastaiagent._internal.errors import LLMProviderError
from fastaiagent.chain.interrupt import Resume, interrupt
from fastaiagent.llm.client import LLMClient
from fastaiagent.llm.message import SystemMessage, UserMessage
from fastaiagent.llm.stream import ReasoningDelta, TextDelta

REASONING_ITEM = {
    "id": "rs_1",
    "type": "reasoning",
    "summary": [{"type": "summary_text", "text": "The user wants Paris weather."}],
    "encrypted_content": "gAAAA-opaque==",
}
CALL_ITEM = {
    "id": "fc_1",
    "type": "function_call",
    "status": "completed",
    "call_id": "call_w1",
    "name": "get_weather",
    "arguments": '{"city": "Paris"}',
}


def _message(text: str) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _response(output: list[dict[str, Any]], status: str = "completed", **extra: Any) -> dict:
    return {
        "id": "resp_1",
        "object": "response",
        "model": "gpt-6-luna",
        "status": status,
        "output": output,
        "usage": {
            "input_tokens": 1000,
            "input_tokens_details": {"cached_tokens": 400, "cache_write_tokens": 0},
            "output_tokens": 50,
            "output_tokens_details": {"reasoning_tokens": 30},
            "total_tokens": 1050,
        },
        **extra,
    }


def get_weather(city: str) -> str:
    """Current weather for a city."""
    return f"{city}: 18C, sunny"


def _client(url: str, **kw: Any) -> LLMClient:
    return LLMClient(
        provider="openai", model="gpt-6-luna", api_key="k", base_url=url, api="responses", **kw
    )


def _sse(*events: dict[str, Any]) -> str:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


# ── the setting ──────────────────────────────────────────────────────────────


def test_api_is_validated_and_only_for_openai_endpoints() -> None:
    with pytest.raises(ValueError, match="api must be one of"):
        LLMClient(provider="openai", api="assistants")
    with pytest.raises(ValueError, match="does not serve it"):
        LLMClient(provider="anthropic", api="responses")
    for provider in ("openai", "azure", "custom"):
        assert LLMClient(provider=provider, model="m", api="responses").api == "responses"


def test_api_survives_to_dict_and_is_absent_on_the_default() -> None:
    assert "api" not in LLMClient(provider="openai", model="gpt-6-luna").to_dict()
    data = LLMClient(provider="openai", model="gpt-6-luna", api="responses").to_dict()
    assert data["api"] == "responses"
    assert LLMClient.from_dict(data).api == "responses"
    agent = Agent(name="a", llm=LLMClient(provider="openai", model="gpt-6-luna", api="responses"))
    assert Agent.from_dict(agent.to_dict()).llm.api == "responses"


# ── one call ─────────────────────────────────────────────────────────────────


def test_request_shape(capture_server) -> None:
    capture_server.set_response_sequence([(200, _response([_message("ok")]))])
    client = _client(capture_server.url, reasoning="medium", max_tokens=300, temperature=0.2)
    client.complete(
        [SystemMessage("Be brief."), UserMessage("Weather?")],
        tools=[FunctionTool(name="get_weather", fn=get_weather).to_openai_format()],
        tool_choice="required",
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "W", "schema": {"type": "object"}, "strict": True},
        },
    )
    (req,) = capture_server.requests
    assert req["path"] == "/responses"
    assert req["headers"]["Authorization"] == "Bearer k"
    body = req["body"]
    assert body["store"] is False
    assert body["input"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Weather?"},
    ]
    (tool,) = body["tools"]
    assert tool["type"] == "function" and tool["name"] == "get_weather"
    assert "function" not in tool  # flat, not Chat's nested shape
    assert body["tool_choice"] == "required"
    assert body["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert body["max_output_tokens"] == 300
    assert body["temperature"] == 0.2
    assert body["text"] == {
        "format": {"type": "json_schema", "name": "W", "schema": {"type": "object"}, "strict": True}
    }
    assert "reasoning.encrypted_content" in body["include"]


def test_response_is_parsed_into_the_same_llm_response(capture_server) -> None:
    capture_server.set_response_sequence([(200, _response([REASONING_ITEM, CALL_ITEM]))])
    response = _client(capture_server.url).complete([UserMessage("Weather in Paris?")])
    (call,) = response.tool_calls
    assert (call.id, call.name, call.arguments) == ("call_w1", "get_weather", {"city": "Paris"})
    assert response.finish_reason == "tool_calls"
    assert response.reasoning == "The user wants Paris weather."
    (block,) = response.reasoning_blocks
    assert block.provider == "openai-responses" and block.data == REASONING_ITEM
    assert response.usage["prompt_tokens"] == 1000
    assert response.usage["cached_tokens"] == 400
    assert response.usage["reasoning_tokens"] == 30


def test_incomplete_and_failed_responses(capture_server) -> None:
    capture_server.set_response_sequence(
        [
            (
                200,
                _response(
                    [_message("partial")],
                    status="incomplete",
                    incomplete_details={"reason": "max_output_tokens"},
                ),
            ),
            (200, _response([], status="failed", error={"code": "server_error", "message": "x"})),
        ]
    )
    client = _client(capture_server.url)
    cut = client.complete([UserMessage("long")])
    assert (cut.content, cut.finish_reason) == ("partial", "length")
    with pytest.raises(LLMProviderError, match="failed"):
        client.complete([UserMessage("again")])


def test_chat_completions_refusal_names_the_way_out(capture_server) -> None:
    refusal = {
        "error": {
            "message": "Function tools with reasoning_effort are not supported for gpt-6-luna "
            "in /v1/chat/completions. To use function tools, use /v1/responses or set "
            "reasoning_effort to 'none'.",
            "type": "invalid_request_error",
        }
    }
    capture_server.set_response_sequence([(400, refusal)])
    chat = LLMClient(
        provider="openai", model="gpt-6-luna", api_key="k", base_url=capture_server.url
    )
    with pytest.raises(LLMProviderError, match=r"api='responses'"):
        chat.complete([UserMessage("hi")], reasoning="medium")


def test_responses_calls_are_priced(capture_server) -> None:
    capture_server.set_response_sequence([(200, _response([_message("ok")]))])
    agent = Agent(name="a", llm=_client(capture_server.url))
    result = agent.run("hi")
    # gpt-6-luna: $0.10 in, $0.01 cached, $0.50 out per 1M (official rates).
    expected = (600 * 0.10 + 400 * 0.01 + 50 * 0.50) / 1_000_000
    assert result.cost == pytest.approx(expected)


# ── the tool loop ───────────────────────────────────────────────────────────


def test_tool_loop_sends_reasoning_items_back_with_the_call(capture_server) -> None:
    capture_server.set_response_sequence(
        [
            (200, _response([REASONING_ITEM, CALL_ITEM])),
            (200, _response([_message("It is 18C and sunny in Paris.")])),
        ]
    )
    agent = Agent(
        name="w",
        llm=_client(capture_server.url, reasoning="medium"),
        tools=[FunctionTool(name="get_weather", fn=get_weather)],
    )
    result = agent.run("Weather in Paris?")
    assert result.output == "It is 18C and sunny in Paris."
    second = capture_server.requests[1]["body"]["input"]
    assert second[1:] == [
        REASONING_ITEM,
        {
            "type": "function_call",
            "call_id": "call_w1",
            "name": "get_weather",
            "arguments": json.dumps({"city": "Paris"}),
        },
        {"type": "function_call_output", "call_id": "call_w1", "output": "Paris: 18C, sunny"},
    ]


def test_reasoning_items_survive_a_pause_and_resume(capture_server, tmp_path) -> None:
    def approve(city: str) -> str:
        """Ask a human before checking."""
        decision = interrupt(reason="ok?", context={"city": city})
        return f"{city}: approved={decision.approved}"

    call = {**CALL_ITEM, "name": "approve"}
    capture_server.set_response_sequence(
        [(200, _response([REASONING_ITEM, call])), (200, _response([_message("done")]))]
    )
    agent = Agent(
        name="w",
        llm=_client(capture_server.url, reasoning="medium"),
        tools=[FunctionTool(name="approve", fn=approve)],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "cp.db")),
    )
    paused = agent.run("Paris?", execution_id="ex-1")
    assert paused.status == "paused"
    resumed = agent.resume("ex-1", resume_value=Resume(approved=True))
    assert resumed.output == "done"
    sent = capture_server.requests[-1]["body"]["input"]
    assert REASONING_ITEM in sent  # restored from the checkpoint, sent back verbatim


@pytest.mark.asyncio
async def test_streamed_tool_loop(capture_server) -> None:
    turn_1 = _sse(
        {"type": "response.created", "response": {"id": "r", "status": "in_progress"}},
        {"type": "response.output_item.added", "item": {**REASONING_ITEM, "summary": []}},
        {"type": "response.reasoning_summary_text.delta", "delta": "Thinking about Paris."},
        {"type": "response.output_item.done", "item": REASONING_ITEM},
        {"type": "response.output_item.added", "item": {**CALL_ITEM, "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "delta": '{"city": "Paris"}'},
        {"type": "response.output_item.done", "item": CALL_ITEM},
        {"type": "response.completed", "response": _response([REASONING_ITEM, CALL_ITEM])},
    )
    turn_2 = _sse(
        {"type": "response.output_text.delta", "delta": "18C "},
        {"type": "response.output_text.delta", "delta": "and sunny."},
        {"type": "response.completed", "response": _response([_message("18C and sunny.")])},
    )
    capture_server.set_response_sequence(
        [(200, turn_1, "text/event-stream"), (200, turn_2, "text/event-stream")]
    )
    agent = Agent(
        name="w",
        llm=_client(capture_server.url, reasoning="low"),
        tools=[FunctionTool(name="get_weather", fn=get_weather)],
    )
    events = [e async for e in agent.astream("Weather in Paris?")]
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "18C and sunny."
    assert any(isinstance(e, ReasoningDelta) and e.text for e in events)
    assert capture_server.requests[0]["body"]["stream"] is True
    second = capture_server.requests[1]["body"]["input"]
    assert REASONING_ITEM in second
    assert {
        "type": "function_call_output",
        "call_id": "call_w1",
        "output": "Paris: 18C, sunny",
    } in second


@pytest.mark.asyncio
async def test_stream_error_event_raises(capture_server) -> None:
    capture_server.set_response_sequence(
        [
            (
                200,
                _sse({"type": "error", "code": "server_error", "message": "boom"}),
                "text/event-stream",
            )
        ]
    )
    with pytest.raises(LLMProviderError, match="boom"):
        async for _ in _client(capture_server.url).astream([UserMessage("hi")]):
            pass


@pytest.mark.asyncio
async def test_injected_openai_client_uses_responses_create(capture_server) -> None:
    openai = pytest.importorskip("openai")
    if not hasattr(openai.AsyncOpenAI(api_key="k", base_url=capture_server.url), "responses"):
        pytest.skip("this openai release has no Responses API")
    capture_server.set_response_sequence([(200, _response([_message("via sdk")]))])
    client = LLMClient(
        provider="openai",
        model="gpt-6-luna",
        api="responses",
        openai_client=openai.AsyncOpenAI(api_key="k", base_url=capture_server.url),
    )
    response = await client.acomplete([UserMessage("hi")])
    assert response.content == "via sdk"
    assert capture_server.requests[0]["path"] == "/responses"
    assert capture_server.requests[0]["body"]["store"] is False
