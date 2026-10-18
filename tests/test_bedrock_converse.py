"""The Bedrock provider speaks the Converse wire format (1.90.0).

``LLMClient(provider="bedrock")`` sent every message in the OpenAI shape
(``{"role": "user", "content": "hi"}``). Converse wants a list of content
blocks, so ``boto3`` refused even a plain "hi" before it left the process
(``ParamValidationError``). And it never sent ``tools`` at all, so a Bedrock
agent could not call one.

Since 1.90.0: text and system messages are content blocks; an assistant's tool
calls are ``toolUse`` blocks and their results ``toolResult`` blocks in a user
turn; same-role neighbours are merged (Converse requires alternation); tools go
in ``toolConfig``; ``tool_choice="required"`` is ``{"any": {}}``. ``base_url``
is boto3's ``endpoint_url`` (a VPC endpoint, or here a test socket).

A real ``boto3`` client — real botocore request building, validation and SigV4
signing — against a real localhost socket (``tests.conftest.CaptureServer``)
answering as Converse does. No AWS account: dummy credentials.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

boto3 = pytest.importorskip("boto3")

from fastaiagent import Agent, AgentConfig, FunctionTool  # noqa: E402
from fastaiagent.llm.client import LLMClient  # noqa: E402
from fastaiagent.llm.message import (  # noqa: E402
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

MODEL = "anthropic.claude-3-haiku-20240307-v1:0"


@pytest.fixture(autouse=True)
def _dummy_aws(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDUMMY")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "dummy-secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def _text(text: str, stop: str = "end_turn") -> dict[str, Any]:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": stop,
        "usage": {"inputTokens": 7, "outputTokens": 3, "totalTokens": 10},
        "metrics": {"latencyMs": 1},
    }


def _tool_use(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "tu-1", "name": name, "input": args}}],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 7, "outputTokens": 3, "totalTokens": 10},
        "metrics": {"latencyMs": 1},
    }


def _client(url: str, **kw: Any) -> LLMClient:
    return LLMClient(provider="bedrock", model=MODEL, base_url=url, **kw)


def _bodies(capture: Any) -> list[dict[str, Any]]:
    return [r["body"] for r in capture.requests if r["path"].endswith("/converse")]


def test_a_plain_message_is_sent_as_content_blocks(capture_server) -> None:
    capture_server.set_response_sequence([(200, _text("hello"))])
    resp = asyncio.run(
        _client(capture_server.url).acomplete([SystemMessage("be brief"), UserMessage("hi")])
    )

    assert resp.content == "hello"
    assert resp.usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    (body,) = _bodies(capture_server)
    assert body["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]
    assert body["system"] == [{"text": "be brief"}]
    from urllib.parse import quote

    # botocore percent-encodes the ':' in the model id, as the Converse API expects.
    assert capture_server.requests[0]["path"] == f"/model/{quote(MODEL, safe='')}/converse"


def test_sampling_options_go_in_inference_config(capture_server) -> None:
    capture_server.set_response_sequence([(200, _text("ok"))])
    asyncio.run(
        _client(capture_server.url, temperature=0.3).acomplete([UserMessage("hi")], max_tokens=50)
    )
    (body,) = _bodies(capture_server)
    assert body["inferenceConfig"]["temperature"] == 0.3
    assert body["inferenceConfig"]["maxTokens"] == 50


def test_same_role_neighbours_are_merged(capture_server) -> None:
    capture_server.set_response_sequence([(200, _text("ok"))])
    asyncio.run(
        _client(capture_server.url).acomplete([UserMessage("first"), UserMessage("second")])
    )
    (body,) = _bodies(capture_server)
    assert body["messages"] == [
        {"role": "user", "content": [{"text": "first"}, {"text": "second"}]}
    ]


def test_a_length_stop_is_reported_as_length(capture_server) -> None:
    capture_server.set_response_sequence([(200, _text("cut", stop="max_tokens"))])
    resp = asyncio.run(_client(capture_server.url).acomplete([UserMessage("hi")]))
    assert resp.finish_reason == "length"


def _lookup(order: str) -> str:
    """Look up an order's status."""
    return f"order {order} shipped"


def test_an_agent_calls_a_tool_through_converse(capture_server) -> None:
    capture_server.set_response_sequence(
        [(200, _tool_use("lookup", {"order": "7"})), (200, _text("It shipped."))]
    )
    agent = Agent(
        name="bedrock-agent",
        system_prompt="Use tools.",
        llm=_client(capture_server.url),
        tools=[FunctionTool(name="lookup", fn=_lookup)],
    )

    result = agent.run("where is order 7?", trace=False)

    assert result.output == "It shipped."
    first, second = _bodies(capture_server)
    spec = first["toolConfig"]["tools"][0]["toolSpec"]
    assert spec["name"] == "lookup"
    assert spec["inputSchema"]["json"]["properties"]["order"]
    assert second["messages"][1] == {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "tu-1", "name": "lookup", "input": {"order": "7"}}}],
    }
    assert second["messages"][2] == {
        "role": "user",
        "content": [
            {"toolResult": {"toolUseId": "tu-1", "content": [{"text": "order 7 shipped"}]}}
        ],
    }
    assert "toolConfig" in second, "Converse requires toolConfig once tool blocks are present"


def test_parallel_tool_results_share_one_user_turn(capture_server) -> None:
    capture_server.set_response_sequence([(200, _text("done"))])
    history = [
        UserMessage("check both"),
        AssistantMessage(
            tool_calls=[
                ToolCall(id="a", name="lookup", arguments={"order": "1"}),
                ToolCall(id="b", name="lookup", arguments={"order": "2"}),
            ]
        ),
        ToolMessage("one", "a"),
        ToolMessage("two", "b"),
    ]
    tools = [FunctionTool(name="lookup", fn=_lookup).to_openai_format()]
    asyncio.run(_client(capture_server.url).acomplete(history, tools=tools))

    (body,) = _bodies(capture_server)
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    ids = [b["toolResult"]["toolUseId"] for b in body["messages"][2]["content"]]
    assert ids == ["a", "b"]


def test_tool_choice_required_is_any(capture_server) -> None:
    capture_server.set_response_sequence(
        [(200, _tool_use("lookup", {"order": "7"})), (200, _text("It shipped."))]
    )
    agent = Agent(
        name="bedrock-required",
        llm=_client(capture_server.url),
        tools=[FunctionTool(name="lookup", fn=_lookup)],
        config=AgentConfig(tool_choice="required"),
    )
    agent.run("hi", trace=False)
    first, second = _bodies(capture_server)
    assert first["toolConfig"]["toolChoice"] == {"any": {}}
    assert "toolChoice" not in second["toolConfig"], "required forces the first turn only"


def test_tool_choice_none_warns_on_bedrock(capture_server, caplog) -> None:
    """Converse has no "none", and requires toolConfig once tool blocks are in
    the history — so it cannot be honoured; it warns rather than pretend."""
    capture_server.set_response_sequence([(200, _text("4"))])
    tools = [FunctionTool(name="lookup", fn=_lookup).to_openai_format()]
    with caplog.at_level(logging.WARNING, logger="fastaiagent"):
        asyncio.run(
            _client(capture_server.url).acomplete(
                [UserMessage("2+2?")], tools=tools, tool_choice="none"
            )
        )
    (body,) = _bodies(capture_server)
    assert "toolChoice" not in body.get("toolConfig", {})
    assert any("tool_choice" in r.getMessage() for r in caplog.records)
