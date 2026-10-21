"""A Gemini agent with ``output_type`` and tools works (1.91.0, X20).

The SDK put Gemini's JSON mode (``responseMimeType: application/json`` plus
``responseSchema``) on every call of an ``output_type`` agent, tool turns
included. Gemini does not take JSON mode together with function calling —
measured 2026-10-19:

* ``gemini-2.5-flash`` refuses the request: HTTP 400 "Function calling with a
  response mime type: 'application/json' is unsupported";
* ``gemini-3.5-flash-lite`` / ``3.8-flash`` answer every tool result with another
  call, so the run ends in ``MaxIterationsError``.

Now, while function tools are in the request, JSON mode is not sent; the system
prompt already asks for JSON, and the reply's code fence is stripped before it is
parsed. A call with no tools (an agent without tools, the structured re-ask) still
gets the native schema.

Real ``Agent`` → real ``LLMClient`` → real ``httpx`` → ``tests.conftest.CaptureServer``
answering as Gemini does.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from fastaiagent import Agent, FunctionTool, LLMClient

USAGE = {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}


class Forecast(BaseModel):
    city: str
    celsius: float


def get_weather(city: str) -> str:
    """Current weather for a city."""
    return "24C, sunny"


def _turn(parts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
        "usageMetadata": USAGE,
    }


CALL = _turn(
    [
        {
            "functionCall": {"name": "get_weather", "args": {"city": "Rome"}},
            "thoughtSignature": "sig",
        }
    ]
)
# What Gemini 3.8 actually returns without JSON mode: fenced JSON.
FENCED = _turn([{"text": '```json\n{"city": "Rome", "celsius": 24}\n```'}])


def _agent(url: str, **kw: Any) -> Agent:
    return Agent(
        name="forecaster",
        llm=LLMClient(provider="gemini", model="gemini-3.8-flash", api_key="k", base_url=url),
        output_type=Forecast,
        **kw,
    )


def _generation_config(request: dict[str, Any]) -> dict[str, Any]:
    return request["body"].get("generationConfig") or {}


def test_a_tool_turn_is_sent_without_json_mode(capture_server) -> None:
    capture_server.set_response_sequence([(200, CALL), (200, FENCED)])
    result = _agent(
        capture_server.url, tools=[FunctionTool(name="get_weather", fn=get_weather)]
    ).run("Rome weather as JSON")
    for request in capture_server.requests:
        assert "tools" in request["body"]
        config = _generation_config(request)
        assert "responseMimeType" not in config and "responseSchema" not in config
    # The prompt still asks for the schema, and the fenced reply parses.
    system = capture_server.requests[0]["body"]["systemInstruction"]["parts"][0]["text"]
    assert "JSON" in system
    assert result.parsed == Forecast(city="Rome", celsius=24)
    assert result.output.strip().startswith("{"), "the fence is stripped from the output too"


def test_a_tool_less_call_keeps_the_native_schema(capture_server) -> None:
    capture_server.set_response_sequence(
        [(200, _turn([{"text": '{"city": "Oslo", "celsius": -4}'}]))]
    )
    result = _agent(capture_server.url).run("Oslo weather")
    config = _generation_config(capture_server.requests[0])
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"]["properties"]["city"]["type"] == "string"
    assert result.parsed == Forecast(city="Oslo", celsius=-4)


def test_a_streamed_tool_turn_is_sent_without_json_mode(capture_server) -> None:
    import json

    def sse(body: dict[str, Any]) -> str:
        return f"data: {json.dumps(body)}\n\n"

    capture_server.set_response_sequence(
        [(200, sse(CALL), "text/event-stream"), (200, sse(FENCED), "text/event-stream")]
    )
    agent = _agent(capture_server.url, tools=[FunctionTool(name="get_weather", fn=get_weather)])
    result = agent.stream("Rome weather as JSON")
    for request in capture_server.requests:
        assert "responseMimeType" not in _generation_config(request)
    assert result.parsed == Forecast(city="Rome", celsius=24)
