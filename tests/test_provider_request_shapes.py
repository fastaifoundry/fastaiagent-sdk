"""Request shapes the real-provider check of 1.92.0 found wrong (1.93.0).

The behaviour is proven live in ``tests/e2e/test_gate_ollama.py`` (Ollama) and
``tests/e2e/test_providers_live.py`` (Groq); these pin the request the SDK
builds, so CI — which has neither — still catches a regression.
"""

from __future__ import annotations

from fastaiagent.llm.client import LLMClient, _apply_ollama_format, _ollama_tool_calls
from fastaiagent.llm.message import AssistantMessage, SystemMessage, ToolCall, UserMessage

SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "CityFact", "schema": {"type": "object"}},
}
TOOLS = [{"type": "function", "function": {"name": "get_population", "parameters": {}}}]


def test_ollama_gets_tool_call_arguments_as_an_object() -> None:
    msg = AssistantMessage(
        content=None, tool_calls=[ToolCall(id="c1", name="lookup", arguments={"city": "Paris"})]
    )
    (call,) = msg.to_provider_dict("ollama")["tool_calls"]
    assert call["function"]["arguments"] == {"city": "Paris"}
    # Every other provider keeps OpenAI's JSON string.
    (call,) = msg.to_provider_dict("openai")["tool_calls"]
    assert call["function"]["arguments"] == '{"city": "Paris"}'


def test_ollama_keeps_its_own_call_ids() -> None:
    msg = {
        "tool_calls": [
            {"id": "call_kay6v1wr", "function": {"name": "a", "arguments": {}}},
            {"function": {"name": "b", "arguments": {"x": 1}}},
        ]
    }
    assert _ollama_tool_calls(msg, 3) == [("call_kay6v1wr", "a", {}), ("call_4", "b", {"x": 1})]


def test_ollama_format_is_sent_only_without_tools() -> None:
    body: dict = {"messages": [{"role": "user", "content": "hi"}]}
    _apply_ollama_format(body, None, SCHEMA)
    assert body["format"] == {"type": "object"}

    body = {"messages": [{"role": "system", "content": "Be brief."}]}
    _apply_ollama_format(body, TOOLS, SCHEMA)
    assert "format" not in body
    system = body["messages"][0]["content"]
    assert system.startswith("Be brief.") and "Call the tools you need first" in system


def _openai_body(provider: str, tools: list | None) -> dict:
    client = LLMClient(provider=provider, model="m", api_key="k")
    body, _ = client._build_openai_body(
        [SystemMessage("Be brief."), UserMessage("Lyon?")], tools, response_format=SCHEMA
    )
    return body


def test_groq_puts_the_schema_in_the_prompt_beside_tools() -> None:
    body = _openai_body("groq", TOOLS)
    assert "response_format" not in body
    assert "Call the tools you need first" in body["messages"][0]["content"]


def test_groq_without_tools_and_openai_keep_the_native_schema() -> None:
    assert _openai_body("groq", None)["response_format"] == SCHEMA
    assert _openai_body("openai", TOOLS)["response_format"] == SCHEMA
