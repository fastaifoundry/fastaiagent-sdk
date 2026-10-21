"""Reasoning control and reasoning output (1.91.0, 2A) — and Claude's thinking blocks
surviving the tool loop (X17).

``LLMClient(reasoning=...)`` takes one vocabulary — ``"none"``, ``"minimal"``,
``"low"``, ``"medium"``, ``"high"``, ``"xhigh"``, ``"max"`` — and puts it in each
provider's own field:

* OpenAI-compatible: ``reasoning_effort``, as given;
* Anthropic: ``output_config.effort`` with adaptive thinking (``display:
  "summarized"`` so the text is returned) on models that have it; extended thinking
  with a ``budget_tokens`` on the ones that do not; ``"none"`` turns thinking off;
* Gemini: ``thinkingLevel`` (3.x / 4) or ``thinkingBudget`` (2.5), with
  ``includeThoughts``;
* Bedrock Claude: the Anthropic fields, as ``additionalModelRequestFields``.

What comes back: ``LLMResponse.reasoning`` (the text), ``reasoning_blocks`` (the
provider's opaque blocks), ``usage["reasoning_tokens"]``, and ``ReasoningDelta``
stream events. Reasoning text is never part of ``content``.

X17: Claude returns thinking blocks — on 5.x models without being asked — and the
API requires them back, unchanged, within a tool-use turn. The SDK dropped them, so
every Claude agent with tools lost its own reasoning between tool calls (the API
does not error; it silently drops the thinking). They now ride on the assistant
message — through ``run``, ``astream``, checkpoints and resume — and go back only to
the provider that issued them.

Real ``LLMClient`` → real ``httpx`` (real ``boto3`` for Bedrock) →
``tests.conftest.CaptureServer`` answering as each provider does.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from fastaiagent import Agent, FunctionTool, SQLiteCheckpointer
from fastaiagent.chain.interrupt import Resume, interrupt
from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.llm.message import (
    AssistantMessage,
    ReasoningBlock,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from fastaiagent.llm.stream import ReasoningDelta, TextDelta

THINK = {"type": "thinking", "thinking": "17 x 23 = 391.", "signature": "sig-abc=="}
REDACTED = {"type": "redacted_thinking", "data": "opaque-redacted=="}


def lookup(order: str) -> str:
    """Look up an order."""
    return f"order {order} shipped"


TOOLS = [FunctionTool(name="lookup", fn=lookup)]


def _anthropic_msg(content: list[dict[str, Any]], stop: str = "end_turn", **usage: Any) -> dict:
    return {
        "id": "m1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": content,
        "stop_reason": stop,
        "usage": {"input_tokens": 5, "output_tokens": 20, **usage},
    }


ANTHROPIC_CALL = _anthropic_msg(
    [THINK, REDACTED, {"type": "tool_use", "id": "tu1", "name": "lookup", "input": {"order": "7"}}],
    stop="tool_use",
    output_tokens_details={"thinking_tokens": 15},
)
ANTHROPIC_ANSWER = _anthropic_msg([{"type": "text", "text": "It shipped."}])

OPENAI_OK = {
    "id": "c1",
    "model": "gpt-5-mini",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "391"}, "finish_reason": "stop"}
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 40,
        "total_tokens": 50,
        "completion_tokens_details": {"reasoning_tokens": 32},
    },
}

GEMINI_THOUGHT = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {"text": "Multiply 17 by 23.", "thought": True},
                    {"text": "391", "thoughtSignature": "gem-sig"},
                ],
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 10,
        "candidatesTokenCount": 3,
        "thoughtsTokenCount": 35,
        "totalTokenCount": 48,
    },
}


def _ask(client: LLMClient, messages: list | None = None, **kw: Any) -> LLMResponse:
    return asyncio.run(client.acomplete(messages or [UserMessage("q")], **kw))


def _body(capture: Any, index: int = 0) -> dict[str, Any]:
    return capture.requests[index]["body"]


def _anthropic(url: str, model: str = "claude-haiku-5-5", **kw: Any) -> LLMClient:
    return LLMClient(provider="anthropic", model=model, api_key="k", base_url=url, **kw)


def _sse_anthropic(*events: tuple[str, dict[str, Any]]) -> str:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


# ── the vocabulary ─────────────────────────────────────────────────────────


def test_an_unknown_level_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="reasoning"):
        LLMClient(provider="openai", model="gpt-5-mini", reasoning="hihg")


def test_a_call_with_an_unknown_level_is_refused(capture_server) -> None:
    client = LLMClient(
        provider="openai", model="gpt-5-mini", api_key="k", base_url=capture_server.url
    )
    with pytest.raises(ValueError, match="reasoning"):
        _ask(client, reasoning="extreme")
    assert capture_server.requests == [], "refused before anything was sent"


# ── what is sent ───────────────────────────────────────────────────────────


def test_openai_sends_reasoning_effort_and_a_call_can_override(capture_server) -> None:
    capture_server.set_response_sequence([(200, OPENAI_OK)] * 3)
    url = capture_server.url
    client = LLMClient(
        provider="openai", model="gpt-5-mini", api_key="k", base_url=url, reasoning="low"
    )
    _ask(client)
    _ask(client, reasoning="xhigh")
    _ask(LLMClient(provider="openai", model="gpt-5-mini", api_key="k", base_url=url))
    assert _body(capture_server, 0)["reasoning_effort"] == "low"
    assert _body(capture_server, 1)["reasoning_effort"] == "xhigh"
    assert "reasoning_effort" not in _body(capture_server, 2), "unset sends nothing"


@pytest.mark.parametrize(
    ("model", "level", "thinking", "effort"),
    [
        # Effort + adaptive thinking; "summarized" so the 5.x models return the text.
        ("claude-haiku-5-5", "high", {"type": "adaptive", "display": "summarized"}, "high"),
        ("claude-opus-5-5", "max", {"type": "adaptive", "display": "summarized"}, "max"),
        ("claude-opus-4-8", "medium", {"type": "adaptive", "display": "summarized"}, "medium"),
        ("claude-sonnet-4-6", "low", {"type": "adaptive", "display": "summarized"}, "low"),
        # Anthropic has no "minimal": the lowest effort it has.
        ("claude-haiku-5-5", "minimal", {"type": "adaptive", "display": "summarized"}, "low"),
        # Off.
        ("claude-haiku-5-5", "none", {"type": "disabled"}, None),
        # Sonnet 5.5's lowest setting: no up-front thinking.
        ("claude-sonnet-5-5", "none", {"type": "between_tools"}, None),
        # Extended thinking only: a budget, no effort (Haiku 4.5 400s on effort).
        ("claude-haiku-4-5", "medium", {"type": "enabled", "budget_tokens": 8192}, None),
        # Opus 4.5 takes both.
        (
            "claude-opus-4-5-20251101",
            "high",
            {"type": "enabled", "budget_tokens": 16384},
            "high",
        ),
    ],
)
def test_anthropic_maps_the_level(capture_server, model, level, thinking, effort) -> None:
    capture_server.set_response_sequence([(200, ANTHROPIC_ANSWER)])
    _ask(_anthropic(capture_server.url, model=model, reasoning=level))
    body = _body(capture_server)
    assert body["thinking"] == thinking
    if effort is None:
        assert "output_config" not in body
    else:
        assert body["output_config"] == {"effort": effort}


def test_anthropic_budget_stays_below_max_tokens(capture_server) -> None:
    capture_server.set_response_sequence([(200, ANTHROPIC_ANSWER)])
    _ask(_anthropic(capture_server.url, model="claude-haiku-4-5", reasoning="max", max_tokens=6000))
    body = _body(capture_server)
    assert body["thinking"]["budget_tokens"] < body["max_tokens"] == 6000


def test_anthropic_unset_sends_no_thinking_fields(capture_server) -> None:
    capture_server.set_response_sequence([(200, ANTHROPIC_ANSWER)])
    _ask(_anthropic(capture_server.url))
    body = _body(capture_server)
    assert "thinking" not in body and "output_config" not in body


@pytest.mark.parametrize(
    ("model", "level", "config"),
    [
        ("gemini-3.8-flash", "low", {"thinkingLevel": "low", "includeThoughts": True}),
        ("gemini-4-argon", "max", {"thinkingLevel": "high", "includeThoughts": True}),
        ("gemini-3.5-flash-lite", "minimal", {"thinkingLevel": "minimal", "includeThoughts": True}),
        ("gemini-2.5-flash", "none", {"thinkingBudget": 0}),
        ("gemini-2.5-flash", "medium", {"thinkingBudget": 8192, "includeThoughts": True}),
    ],
)
def test_gemini_maps_the_level(capture_server, model, level, config) -> None:
    capture_server.set_response_sequence([(200, GEMINI_THOUGHT)])
    client = LLMClient(
        provider="gemini", model=model, api_key="k", base_url=capture_server.url, reasoning=level
    )
    _ask(client)
    assert _body(capture_server)["generationConfig"]["thinkingConfig"] == config


def test_gemini_3_cannot_turn_thinking_off_and_says_so(capture_server, caplog) -> None:
    capture_server.set_response_sequence([(200, GEMINI_THOUGHT)])
    client = LLMClient(
        provider="gemini",
        model="gemini-3.6-flash",
        api_key="k",
        base_url=capture_server.url,
        reasoning="none",
    )
    with caplog.at_level("WARNING"):
        _ask(client)
    assert _body(capture_server)["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "minimal"
    }
    assert "cannot turn thinking off" in caplog.text


# ── what comes back ────────────────────────────────────────────────────────


def test_anthropic_reasoning_is_parsed_and_kept_out_of_content(capture_server) -> None:
    capture_server.set_response_sequence(
        [
            (
                200,
                _anthropic_msg(
                    [THINK, {"type": "text", "text": "391"}],
                    output_tokens_details={"thinking_tokens": 15},
                ),
            )
        ]
    )
    response = _ask(_anthropic(capture_server.url, reasoning="low"))
    assert response.content == "391"
    assert response.reasoning == "17 x 23 = 391."
    assert response.reasoning_blocks == [ReasoningBlock(provider="anthropic", data=THINK)]
    assert response.usage["reasoning_tokens"] == 15
    assert response.usage["completion_tokens"] == 20, "thinking is billed inside output"


def test_openai_reports_reasoning_tokens(capture_server) -> None:
    capture_server.set_response_sequence([(200, OPENAI_OK)])
    client = LLMClient(
        provider="openai", model="gpt-5-mini", api_key="k", base_url=capture_server.url
    )
    response = _ask(client)
    assert response.usage["reasoning_tokens"] == 32
    assert response.reasoning is None, "Chat Completions returns no reasoning text"


def test_an_openai_compatible_reasoning_field_is_read(capture_server) -> None:
    """DeepSeek, Groq, OpenRouter and vLLM return the text beside the answer."""
    body = json.loads(json.dumps(OPENAI_OK))
    body["choices"][0]["message"]["reasoning_content"] = "Multiply."
    capture_server.set_response_sequence([(200, body)])
    client = LLMClient(
        provider="deepseek", model="deepseek-reasoner", api_key="k", base_url=capture_server.url
    )
    response = _ask(client)
    assert response.reasoning == "Multiply." and response.content == "391"


def test_gemini_thought_summaries_are_reasoning_not_content(capture_server) -> None:
    capture_server.set_response_sequence([(200, GEMINI_THOUGHT)])
    client = LLMClient(
        provider="gemini",
        model="gemini-3.8-flash",
        api_key="k",
        base_url=capture_server.url,
        reasoning="low",
    )
    response = _ask(client)
    assert response.content == "391"
    assert response.reasoning == "Multiply 17 by 23."
    assert response.usage["reasoning_tokens"] == 35
    assert response.usage["completion_tokens"] == 38


# ── X17: Claude's thinking goes back within the tool loop ──────────────────


def _agent(url: str, **kw: Any) -> Agent:
    return Agent(
        name="claude-think",
        llm=_anthropic(url, model=kw.pop("model", "claude-haiku-5-5")),
        tools=kw.pop("tools", TOOLS),
        **kw,
    )


def _assistant_turn(request: dict[str, Any]) -> list[dict[str, Any]]:
    (turn,) = [m for m in request["body"]["messages"] if m["role"] == "assistant"]
    return turn["content"]


def _assert_thinking_sent_back(request: dict[str, Any]) -> None:
    content = _assistant_turn(request)
    # Unchanged and in the order the model produced them, ahead of the call.
    assert content[0] == THINK
    assert content[1] == REDACTED
    assert content[2]["type"] == "tool_use" and content[2]["id"] == "tu1"


def test_run_sends_claudes_thinking_back(capture_server) -> None:
    """Without ``reasoning=`` set: Claude 5.x thinks by default."""
    capture_server.set_response_sequence([(200, ANTHROPIC_CALL), (200, ANTHROPIC_ANSWER)])
    result = _agent(capture_server.url).run("Where is order 7?")
    assert result.output == "It shipped."
    _assert_thinking_sent_back(capture_server.requests[1])


def test_astream_sends_claudes_thinking_back_and_streams_it(capture_server) -> None:
    stream = _sse_anthropic(
        ("message_start", {"message": {"usage": {"input_tokens": 5}}}),
        (
            "content_block_start",
            {"index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        ),
        (
            "content_block_delta",
            {"index": 0, "delta": {"type": "thinking_delta", "thinking": "17 x 23 "}},
        ),
        (
            "content_block_delta",
            {"index": 0, "delta": {"type": "thinking_delta", "thinking": "= 391."}},
        ),
        (
            "content_block_delta",
            {"index": 0, "delta": {"type": "signature_delta", "signature": "sig-abc=="}},
        ),
        ("content_block_stop", {"index": 0}),
        ("content_block_start", {"index": 1, "content_block": REDACTED}),
        ("content_block_stop", {"index": 1}),
        (
            "content_block_start",
            {
                "index": 2,
                "content_block": {"type": "tool_use", "id": "tu1", "name": "lookup", "input": {}},
            },
        ),
        (
            "content_block_delta",
            {"index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"order": "7"}'}},
        ),
        ("content_block_stop", {"index": 2}),
        ("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 20}}),
    )
    answer = _sse_anthropic(
        ("message_start", {"message": {"usage": {"input_tokens": 9}}}),
        ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
        (
            "content_block_delta",
            {"index": 0, "delta": {"type": "text_delta", "text": "It shipped."}},
        ),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}}),
    )
    capture_server.set_response_sequence(
        [(200, stream, "text/event-stream"), (200, answer, "text/event-stream")]
    )

    async def drain() -> list[Any]:
        return [e async for e in _agent(capture_server.url).astream("Where is order 7?")]

    events = asyncio.run(drain())
    reasoning = "".join(e.text for e in events if isinstance(e, ReasoningDelta))
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert reasoning == "17 x 23 = 391." and text == "It shipped."
    _assert_thinking_sent_back(capture_server.requests[1])


def test_a_resumed_run_still_sends_claudes_thinking(capture_server, tmp_path) -> None:
    def approve(order: str) -> str:
        """Needs approval."""
        interrupt(reason="approve?", context={"order": order})
        return f"order {order} approved"

    call = json.loads(json.dumps(ANTHROPIC_CALL))
    call["content"][2]["name"] = "approve"
    capture_server.set_response_sequence([(200, call), (200, ANTHROPIC_ANSWER)])
    store = SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))
    agent = _agent(
        capture_server.url, tools=[FunctionTool(name="approve", fn=approve)], checkpointer=store
    )
    assert agent.run("Approve order 7.", execution_id="think-resume").status == "paused"
    assert agent.resume("think-resume", resume_value=Resume(approved=True)).status == "completed"
    content = _assistant_turn(capture_server.requests[-1])
    assert content[:2] == [THINK, REDACTED]


def test_blocks_go_back_only_to_the_provider_that_issued_them(capture_server) -> None:
    """An OpenAI or Gemini model cannot read Claude's blocks; sending them is an error."""
    history = [
        UserMessage("go"),
        AssistantMessage(
            tool_calls=[ToolCall(id="tu1", name="lookup", arguments={"order": "7"})],
            reasoning_blocks=[ReasoningBlock(provider="anthropic", data=THINK)],
        ),
        ToolMessage("order 7 shipped", "tu1"),
    ]
    capture_server.set_response_sequence([(200, OPENAI_OK)])
    client = LLMClient(
        provider="openai", model="gpt-5-mini", api_key="k", base_url=capture_server.url
    )
    _ask(client, history, tools=[TOOLS[0].to_openai_format()])
    sent = json.dumps(_body(capture_server))
    assert "sig-abc" not in sent and "thinking" not in sent


def test_a_plain_checkpoint_has_no_reasoning_key(tmp_path) -> None:
    """Dropped when empty, so checkpoints from every provider that issues none —
    which replicate to the plane — keep their exact shape."""
    from fastaiagent.agent.executor import _serialize_messages

    (dumped,) = _serialize_messages([AssistantMessage(content="hi")])
    assert "reasoning_blocks" not in dumped
    (kept,) = _serialize_messages(
        [
            AssistantMessage(
                content="hi", reasoning_blocks=[ReasoningBlock(provider="anthropic", data=THINK)]
            )
        ]
    )
    assert kept["reasoning_blocks"] == [{"provider": "anthropic", "data": THINK}]


# ── Bedrock (real boto3) ───────────────────────────────────────────────────

BEDROCK_MODEL = "us.anthropic.claude-haiku-5-5-v1:0"


@pytest.fixture
def _dummy_aws(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDUMMY")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "dummy-secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


BEDROCK_REASONING = {
    "reasoningContent": {"reasoningText": {"text": "Check it.", "signature": "bsig"}}
}


def _bedrock_turn(content: list[dict[str, Any]], stop: str) -> dict[str, Any]:
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop,
        "usage": {"inputTokens": 7, "outputTokens": 3, "totalTokens": 10},
        "metrics": {"latencyMs": 1},
    }


def test_bedrock_claude_gets_the_fields_and_the_reasoning_goes_back(
    capture_server, _dummy_aws
) -> None:
    pytest.importorskip("boto3")
    call = _bedrock_turn(
        [
            BEDROCK_REASONING,
            {"toolUse": {"toolUseId": "tu-1", "name": "lookup", "input": {"order": "7"}}},
        ],
        "tool_use",
    )
    capture_server.set_response_sequence(
        [(200, call), (200, _bedrock_turn([{"text": "It shipped."}], "end_turn"))]
    )
    agent = Agent(
        name="bedrock-think",
        llm=LLMClient(
            provider="bedrock", model=BEDROCK_MODEL, base_url=capture_server.url, reasoning="low"
        ),
        tools=TOOLS,
    )
    assert agent.run("where is order 7?", trace=False).output == "It shipped."
    first, second = (r["body"] for r in capture_server.requests)
    assert first["additionalModelRequestFields"] == {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
    }
    assert second["messages"][1]["content"][0] == BEDROCK_REASONING
