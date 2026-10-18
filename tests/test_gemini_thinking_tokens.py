"""Gemini thinking tokens are completion tokens, and are priced (1.90.0, F7).

A Gemini 2.5 / 3 thinking model reports ``usageMetadata.thoughtsTokenCount``
next to ``candidatesTokenCount``. Google bills thinking at the output rate, and
``totalTokenCount`` includes it. The provider mapped ``completion_tokens`` from
``candidatesTokenCount`` alone, so a run's cost (priced on prompt + completion)
left the thinking out while ``tokens_used`` (from ``total_tokens``) counted it —
and ``astream``, which reports only prompt + completion, disagreed with ``arun``.

Real ``LLMClient`` → real ``httpx`` → ``tests.conftest.CaptureServer`` answering
as the Gemini API does.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from fastaiagent import Agent
from fastaiagent.llm.client import LLMClient
from fastaiagent.llm.message import UserMessage
from fastaiagent.llm.stream import Usage

USAGE = {
    "promptTokenCount": 100,
    "candidatesTokenCount": 40,
    "thoughtsTokenCount": 900,
    "totalTokenCount": 1040,
}
REPLY = {
    "candidates": [{"content": {"parts": [{"text": "42"}]}, "finishReason": "STOP"}],
    "usageMetadata": USAGE,
    "modelVersion": "gemini-2.5-flash",
}


def _client(url: str) -> LLMClient:
    return LLMClient(provider="gemini", model="gemini-2.5-flash", api_key="k", base_url=url)


def test_completion_tokens_include_thinking(capture_server) -> None:
    capture_server.set_response_sequence([(200, REPLY)])
    resp = asyncio.run(_client(capture_server.url).acomplete([UserMessage("q")]))
    assert resp.usage["completion_tokens"] == 40 + 900
    assert (
        resp.usage["prompt_tokens"] + resp.usage["completion_tokens"] == resp.usage["total_tokens"]
    )


def test_a_reply_without_thinking_is_unchanged(capture_server) -> None:
    plain = dict(
        REPLY,
        usageMetadata={"promptTokenCount": 100, "candidatesTokenCount": 40, "totalTokenCount": 140},
    )
    capture_server.set_response_sequence([(200, plain)])
    resp = asyncio.run(_client(capture_server.url).acomplete([UserMessage("q")]))
    assert resp.usage["completion_tokens"] == 40


def test_the_stream_reports_the_same_tokens(capture_server) -> None:
    chunks = [
        {"candidates": [{"content": {"parts": [{"text": "4"}]}}]},
        {
            "candidates": [{"content": {"parts": [{"text": "2"}]}, "finishReason": "STOP"}],
            "usageMetadata": USAGE,
        },
    ]
    sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    capture_server.set_response_sequence([(200, sse, "text/event-stream")])

    async def drain() -> list[Usage]:
        return [
            e
            async for e in _client(capture_server.url).astream([UserMessage("q")])
            if isinstance(e, Usage)
        ]

    usage = asyncio.run(drain())
    assert usage and usage[-1].completion_tokens == 40 + 900
    assert usage[-1].prompt_tokens == 100


def test_run_cost_prices_the_thinking(capture_server) -> None:
    from fastaiagent._internal.pricing import compute_cost_usd

    capture_server.set_response_sequence([(200, REPLY)])
    result = Agent(name="g", llm=_client(capture_server.url)).run("q", trace=False)

    expected = compute_cost_usd("gemini-2.5-flash", 100, 40 + 900)
    assert expected, "gemini-2.5-flash must be priced for this test to mean anything"
    assert result.tokens_used == 1040
    assert result.cost == pytest.approx(expected), "thinking tokens were not priced"
