"""LLMClient pass-through options, Retry-After, concurrency and fallback models (1.91.0, 2K + 2E).

* ``extra_body`` / ``extra_headers`` reach every provider's request — on the client
  or per call — and never ``to_dict()`` (a header often carries a credential).
  Bedrock takes ``extra_body`` as ``additionalModelRequestFields``.
* A 429 is an :class:`LLMRateLimitError` (still an ``LLMProviderError``) carrying
  ``retry_after`` from the response, and the retry waits that long instead of the
  blind exponential backoff.
* ``max_concurrency`` caps a client's in-flight calls.
* ``fallbacks=[...]``: after a client's own retries fail with a 429, a 5xx, a
  connection error, a timeout or a context overflow, the next client is tried —
  each attempt its own ``llm.*`` span and its own cost. A stream falls back only
  before its first delta. Fallbacks are local (not in ``to_dict()``).

Real ``LLMClient`` → real ``httpx`` → ``tests.conftest.CaptureServer`` answering as
each provider does; the assertions read what was actually sent.
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
from typing import Any

import pytest

from fastaiagent._internal.errors import LLMProviderError, LLMRateLimitError
from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.llm.message import UserMessage
from fastaiagent.llm.stream import TextDelta
from tests.conftest import CaptureServer

OPENAI_OK = {
    "id": "c1",
    "model": "gpt-5-mini",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}
ANTHROPIC_OK = {
    "id": "m1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 2},
}
GEMINI_OK = {
    "candidates": [
        {"content": {"role": "model", "parts": [{"text": "hi"}]}, "finishReason": "STOP"}
    ],
    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7},
}


def _openai(url: str, **kw: Any) -> LLMClient:
    return LLMClient(provider="openai", model="gpt-5-mini", api_key="sk-test", base_url=url, **kw)


def _ask(client: LLMClient, **kw: Any) -> LLMResponse:
    return asyncio.run(client.acomplete([UserMessage("q")], **kw))


def _closed_port_url() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}/v1"


# ── 2K: extra_body / extra_headers ─────────────────────────────────────────


def test_openai_gets_extra_body_and_headers_and_a_call_can_override(capture_server) -> None:
    capture_server.set_response_sequence([(200, OPENAI_OK), (200, OPENAI_OK)])
    client = _openai(
        capture_server.url,
        extra_body={"service_tier": "flex", "metadata": {"team": "a"}},
        extra_headers={"X-Team": "billing"},
    )
    _ask(client)
    _ask(client, extra_body={"service_tier": "priority"}, extra_headers={"X-Trace": "t1"})
    first, second = capture_server.requests
    assert first["body"]["service_tier"] == "flex" and first["body"]["metadata"] == {"team": "a"}
    assert first["headers"]["X-Team"] == "billing"
    assert second["body"]["service_tier"] == "priority"
    assert second["body"]["metadata"] == {"team": "a"}
    assert second["headers"]["X-Team"] == "billing" and second["headers"]["X-Trace"] == "t1"


def test_extra_body_cannot_replace_the_conversation(capture_server) -> None:
    """The messages, model and tools the SDK built win — an extra is extra."""
    capture_server.set_response_sequence([(200, OPENAI_OK)])
    _ask(_openai(capture_server.url, extra_body={"model": "other", "messages": []}))
    body = capture_server.requests[0]["body"]
    assert body["model"] == "gpt-5-mini" and body["messages"]


def test_anthropic_gets_extra_body_and_headers(capture_server) -> None:
    capture_server.set_response_sequence([(200, ANTHROPIC_OK)])
    client = LLMClient(
        provider="anthropic",
        model="claude-haiku-4-5",
        api_key="k",
        base_url=capture_server.url,
        extra_body={"metadata": {"user_id": "u-1"}},
        extra_headers={"anthropic-beta": "output-300k-2026-03-24"},
    )
    _ask(client)
    req = capture_server.requests[0]
    assert req["body"]["metadata"] == {"user_id": "u-1"}
    assert req["headers"]["anthropic-beta"] == "output-300k-2026-03-24"


def test_gemini_gets_extra_body_and_headers(capture_server) -> None:
    capture_server.set_response_sequence([(200, GEMINI_OK)])
    client = LLMClient(
        provider="gemini",
        model="gemini-3.8-flash",
        api_key="k",
        base_url=capture_server.url,
        extra_body={"labels": {"team": "a"}},
        extra_headers={"X-Team": "billing"},
    )
    _ask(client)
    req = capture_server.requests[0]
    assert req["body"]["labels"] == {"team": "a"}
    assert req["headers"]["X-Team"] == "billing"


def test_streams_get_them_too(capture_server) -> None:
    sse = (
        'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )
    capture_server.set_response_sequence([(200, sse, "text/event-stream")])
    client = _openai(
        capture_server.url, extra_body={"service_tier": "flex"}, extra_headers={"X-T": "1"}
    )

    async def drain() -> None:
        async for _ in client.astream([UserMessage("q")]):
            pass

    asyncio.run(drain())
    req = capture_server.requests[0]
    assert req["body"]["service_tier"] == "flex" and req["headers"]["X-T"] == "1"


def test_extras_never_reach_to_dict() -> None:
    client = LLMClient(
        provider="openai",
        model="gpt-5-mini",
        api_key="k",
        extra_body={"service_tier": "flex"},
        extra_headers={"Authorization": "Bearer SECRET"},
        max_concurrency=3,
        fallbacks=[LLMClient(provider="openai", model="gpt-4o-mini", api_key="k")],
    )
    dumped = json.dumps(client.to_dict())
    for word in ("extra_body", "extra_headers", "SECRET", "flex", "max_concurrency", "fallbacks"):
        assert word not in dumped


def test_bedrock_takes_extra_body_as_additional_model_request_fields(
    capture_server, monkeypatch
) -> None:
    pytest.importorskip("boto3")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDUMMY")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "dummy-secret")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    capture_server.set_response_sequence(
        [
            (
                200,
                {
                    "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7},
                    "metrics": {"latencyMs": 1},
                },
            )
        ]
    )
    client = LLMClient(
        provider="bedrock",
        model="anthropic.claude-3-haiku-20240307-v1:0",
        base_url=capture_server.url,
        extra_body={"top_k": 40},
    )
    _ask(client)
    body = capture_server.requests[0]["body"]
    assert body["additionalModelRequestFields"] == {"top_k": 40}


# ── 2K: 429 and Retry-After ────────────────────────────────────────────────


def test_a_429_is_a_rate_limit_error_with_retry_after(capture_server) -> None:
    capture_server.set_response_sequence(
        [(429, {"error": {"message": "slow down"}}, "application/json", {"Retry-After": "7"})]
    )
    with pytest.raises(LLMRateLimitError) as info:
        _ask(_openai(capture_server.url))
    assert isinstance(info.value, LLMProviderError)  # existing handlers still catch it
    assert info.value.status_code == 429 and info.value.retry_after == 7.0


def test_retry_waits_what_the_provider_asked_not_the_blind_backoff(capture_server) -> None:
    """``Retry-After: 0`` → no wait; the blind first backoff is a full second."""
    capture_server.set_response_sequence(
        [
            (429, {"error": {"message": "slow"}}, "application/json", {"Retry-After": "0"}),
            (200, OPENAI_OK),
        ]
    )
    started = time.monotonic()
    assert _ask(_openai(capture_server.url, max_retries=1)).content == "hi"
    assert time.monotonic() - started < 0.8


def test_retry_after_ms_is_read_too(capture_server) -> None:
    capture_server.set_response_sequence(
        [(429, {"error": {"message": "slow"}}, "application/json", {"retry-after-ms": "1500"})]
    )
    with pytest.raises(LLMRateLimitError) as info:
        _ask(_openai(capture_server.url))
    assert info.value.retry_after == 1.5


# ── 2K: max_concurrency ────────────────────────────────────────────────────


class _Slow(LLMClient):
    """A real LLMClient whose provider call is a timed sleep, so concurrency can
    be counted without a network."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(provider="openai", model="gpt-5-mini", api_key="k", **kw)
        self.active = 0
        self.peak = 0

    def _get_provider_fn(self):
        async def call(messages, tools=None, **kwargs):
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.05)
            self.active -= 1
            return LLMResponse(content="ok", usage={"prompt_tokens": 1, "completion_tokens": 1})

        return call


@pytest.mark.parametrize(("cap", "peak"), [(None, 5), (2, 2), (1, 1)])
def test_max_concurrency_caps_in_flight_calls(cap: int | None, peak: int) -> None:
    client = _Slow(max_concurrency=cap)

    async def burst() -> None:
        await asyncio.gather(*[client.acomplete([UserMessage("q")]) for _ in range(5)])

    asyncio.run(burst())
    asyncio.run(burst())  # a second event loop: the cap must still work
    assert client.peak == peak


def test_max_concurrency_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        LLMClient(provider="openai", model="m", api_key="k", max_concurrency=0)


# ── 2E: fallbacks ──────────────────────────────────────────────────────────


@pytest.fixture
def backup() -> Any:
    server = CaptureServer().start()
    try:
        yield server
    finally:
        server.stop()


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (503, {"error": {"message": "overloaded"}}),
        (429, {"error": {"message": "rate limited"}}),
        (400, {"error": {"message": "too long", "code": "context_length_exceeded"}}),
    ],
    ids=["5xx", "429", "context-overflow"],
)
def test_a_failing_model_falls_back(status: int, body: dict, capture_server, backup) -> None:
    capture_server.set_response_sequence([(status, body)])
    backup.set_response_sequence([(200, {**OPENAI_OK, "model": "gpt-4o-mini"})])
    client = _openai(
        capture_server.url,
        fallbacks=[
            LLMClient(provider="openai", model="gpt-4o-mini", api_key="k", base_url=backup.url)
        ],
    )
    response = _ask(client)
    assert response.content == "hi" and response.model == "gpt-4o-mini"
    assert len(capture_server.requests) == 1 and len(backup.requests) == 1


def test_a_connection_error_falls_back(backup) -> None:
    backup.set_response_sequence([(200, OPENAI_OK)])
    client = _openai(
        _closed_port_url(),
        fallbacks=[
            LLMClient(provider="openai", model="gpt-4o-mini", api_key="k", base_url=backup.url)
        ],
    )
    assert _ask(client).content == "hi"


def test_a_bad_request_does_not_fall_back(capture_server, backup) -> None:
    """A 400 that is not a context overflow is the caller's bug; another model
    would fail the same way, and trying it hides the bug."""
    capture_server.set_response_sequence([(400, {"error": {"message": "bad param"}})])
    client = _openai(
        capture_server.url,
        fallbacks=[
            LLMClient(provider="openai", model="gpt-4o-mini", api_key="k", base_url=backup.url)
        ],
    )
    with pytest.raises(LLMProviderError):
        _ask(client)
    assert backup.requests == []


def test_own_retries_run_before_the_fallback(capture_server, backup) -> None:
    capture_server.set_response_sequence(
        [
            (503, {"error": {"message": "x"}}, "application/json", {"Retry-After": "0"}),
            (503, {"error": {"message": "x"}}, "application/json", {"Retry-After": "0"}),
        ]
    )
    backup.set_response_sequence([(200, OPENAI_OK)])
    client = _openai(
        capture_server.url,
        max_retries=1,
        fallbacks=[
            LLMClient(provider="openai", model="gpt-4o-mini", api_key="k", base_url=backup.url)
        ],
    )
    assert _ask(client).content == "hi"
    assert len(capture_server.requests) == 2


def test_the_last_error_raises_when_every_model_fails(capture_server, backup) -> None:
    capture_server.set_response_sequence([(503, {"error": {"message": "primary down"}})])
    backup.set_response_sequence([(503, {"error": {"message": "backup down"}})])
    client = _openai(
        capture_server.url,
        fallbacks=[
            LLMClient(provider="openai", model="gpt-4o-mini", api_key="k", base_url=backup.url)
        ],
    )
    with pytest.raises(LLMProviderError, match="backup down"):
        _ask(client)


def test_a_stream_falls_back_before_its_first_delta(capture_server, backup) -> None:
    capture_server.set_response_sequence([(503, {"error": {"message": "overloaded"}})])
    sse = (
        'data: {"choices":[{"index":0,"delta":{"content":"from backup"}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )
    backup.set_response_sequence([(200, sse, "text/event-stream")])
    client = _openai(
        capture_server.url,
        fallbacks=[
            LLMClient(provider="openai", model="gpt-4o-mini", api_key="k", base_url=backup.url)
        ],
    )

    async def drain() -> str:
        return "".join(
            [e.text async for e in client.astream([UserMessage("q")]) if isinstance(e, TextDelta)]
        )

    assert asyncio.run(drain()) == "from backup"


class _BreaksMidStream(LLMClient):
    async def _stream_openai(self, messages, tools=None, **kwargs):
        yield TextDelta(text="partial ")
        raise LLMProviderError("connection reset mid-stream", status_code=503)


def test_a_stream_that_fails_after_a_delta_does_not_fall_back(backup) -> None:
    """The caller already has part of one model's answer; splicing another
    model's answer after it would be worse than the error."""
    client = _BreaksMidStream(
        provider="openai",
        model="gpt-5-mini",
        api_key="k",
        fallbacks=[
            LLMClient(provider="openai", model="gpt-4o-mini", api_key="k", base_url=backup.url)
        ],
    )

    async def drain() -> list[str]:
        return [
            e.text async for e in client.astream([UserMessage("q")]) if isinstance(e, TextDelta)
        ]

    with pytest.raises(LLMProviderError, match="mid-stream"):
        asyncio.run(drain())
    assert backup.requests == []


def test_each_attempt_is_its_own_span_and_cost_follows_the_model_used(
    capture_server, backup, monkeypatch, tmp_path
) -> None:
    from fastaiagent import Agent
    from fastaiagent._internal import instance as _instance
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel
    from fastaiagent.trace.otel import get_tracer_provider
    from fastaiagent.trace.storage import TraceStore

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    reset_config()
    _project.set_project_id("test-proj")
    _instance.reset_for_testing()
    otel.reset()
    try:
        capture_server.set_response_sequence([(503, {"error": {"message": "down"}})])
        backup.set_response_sequence([(200, {**OPENAI_OK, "model": "gpt-4o-mini"})])
        llm = _openai(
            capture_server.url,
            fallbacks=[
                LLMClient(provider="openai", model="gpt-4o-mini", api_key="k", base_url=backup.url)
            ],
        )
        result = Agent(name="fb", llm=llm).run("q")
        get_tracer_provider().force_flush()
        names = [s.name for s in TraceStore(str(db)).get_trace(result.trace_id).spans]
        assert "llm.openai.gpt-5-mini" in names and "llm.openai.gpt-4o-mini" in names
        # Priced as gpt-4o-mini (the model that answered): 5 in, 2 out.
        assert result.cost_known
        assert result.cost == pytest.approx((5 * 0.15 + 2 * 0.60) / 1_000_000)
    finally:
        otel.reset()
        _instance.reset_for_testing()
        _project.reset_for_testing()
        reset_config()
