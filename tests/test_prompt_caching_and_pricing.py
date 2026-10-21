"""Prompt caching, cache-aware cost, and a current price table (1.91.0, 2B + X15 + X18).

``LLMClient(cache=True)`` (or ``"1h"``) asks the providers that need asking:
Anthropic gets its automatic top-level ``cache_control``; Bedrock gets ``cachePoint``
blocks after the system prompt, after the tools and on the last turn. OpenAI and
Gemini cache on their own, so nothing is sent for them.

Every provider's cache counts are read into one shape — ``usage["cached_tokens"]``
(read from the cache) and ``usage["cache_write_tokens"]`` (written to it) — and
``prompt_tokens`` is always the whole prompt: Anthropic and Bedrock report the
uncached part only, so their cache tokens are added back.

The cost of a call prices each part at its own rate (X18): before, every prompt
token was billed at the full input rate, so a cache hit was over-reported (OpenAI
reads cost 1/10) and a GPT-6 cache write (1.25x) under-reported. Long-context tiers
apply above their threshold. And the table is current (X15): the GPT-6 family was
unpriced, Opus 5.5 / Sonnet 5.5 took the 5.0 rates, and the ``gpt-5`` row priced
every later ``gpt-5.x`` (gpt-5.5 is 4x it). A built-in row no longer covers a later
version of its model — an unknown model is unpriced, never mispriced.

Reasoning tokens are already inside every provider's completion count, so they are
billed at the output rate with no change.

Real ``LLMClient`` → real ``httpx`` (real ``boto3``) → ``tests.conftest.CaptureServer``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date
from typing import Any

import pytest

from fastaiagent import Agent, FunctionTool
from fastaiagent._internal.pricing import (
    compute_cost_usd,
    reload_rate_overrides,
    set_rate_overrides,
    usage_cost,
)
from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.llm.message import SystemMessage, UserMessage
from fastaiagent.llm.stream import Usage

M = 1_000_000
#: A prompt below every long-context threshold.
STANDARD = 100_000


@pytest.fixture(autouse=True)
def _no_overrides():
    set_rate_overrides(None)
    yield
    set_rate_overrides(None)


def _ask(client: LLMClient, messages: list | None = None, **kw: Any) -> LLMResponse:
    return asyncio.run(client.acomplete(messages or [UserMessage("q")], **kw))


def _anthropic(url: str, model: str = "claude-haiku-5-5", **kw: Any) -> LLMClient:
    return LLMClient(provider="anthropic", model=model, api_key="k", base_url=url, **kw)


ANTHROPIC_CACHED = {
    "id": "m1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": "391"}],
    "stop_reason": "end_turn",
    # As measured live 2026-10-18 on claude-haiku-5-5: input_tokens is only what
    # follows the cache breakpoint.
    "usage": {
        "input_tokens": 4,
        "cache_creation_input_tokens": 4829,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 4829, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 21,
    },
}

OPENAI_CACHED = {
    "id": "c1",
    "model": "gpt-6-luna",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "391"}, "finish_reason": "stop"}
    ],
    # Measured live on gpt-6-luna: the first call writes, the second reads.
    "usage": {
        "prompt_tokens": 3629,
        "completion_tokens": 4,
        "total_tokens": 3633,
        "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 3626},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
}


# ── what is sent ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("setting", "control"),
    [(True, {"type": "ephemeral"}), ("1h", {"type": "ephemeral", "ttl": "1h"})],
)
def test_anthropic_cache_sends_automatic_cache_control(capture_server, setting, control) -> None:
    capture_server.set_response_sequence([(200, ANTHROPIC_CACHED)])
    _ask(_anthropic(capture_server.url, cache=setting), [SystemMessage("rules"), UserMessage("q")])
    assert capture_server.requests[0]["body"]["cache_control"] == control


def test_anthropic_without_cache_sends_nothing(capture_server) -> None:
    capture_server.set_response_sequence([(200, ANTHROPIC_CACHED)])
    _ask(_anthropic(capture_server.url))
    assert "cache_control" not in capture_server.requests[0]["body"]


def test_an_unknown_cache_setting_is_refused() -> None:
    with pytest.raises(ValueError, match="cache"):
        LLMClient(provider="anthropic", model="claude-haiku-5-5", cache="forever")


def test_openai_sends_nothing_for_cache(capture_server) -> None:
    capture_server.set_response_sequence([(200, OPENAI_CACHED)])
    client = LLMClient(
        provider="openai", model="gpt-6-luna", api_key="k", base_url=capture_server.url, cache=True
    )
    _ask(client)
    body = capture_server.requests[0]["body"]
    assert "cache_control" not in body and "prompt_cache_retention" not in body


# ── what comes back ────────────────────────────────────────────────────────


def test_anthropic_cache_counts_and_the_whole_prompt(capture_server) -> None:
    capture_server.set_response_sequence([(200, ANTHROPIC_CACHED)])
    usage = _ask(_anthropic(capture_server.url, cache=True)).usage
    assert usage["prompt_tokens"] == 4 + 4829
    assert usage["cache_write_tokens"] == 4829
    assert usage["cached_tokens"] == 0
    assert usage["total_tokens"] == 4 + 4829 + 21


def test_openai_cache_counts_are_flattened(capture_server) -> None:
    capture_server.set_response_sequence([(200, OPENAI_CACHED)])
    client = LLMClient(
        provider="openai", model="gpt-6-luna", api_key="k", base_url=capture_server.url
    )
    usage = _ask(client).usage
    assert usage["prompt_tokens"] == 3629, "OpenAI's prompt already includes the cache"
    assert usage["cache_write_tokens"] == 3626 and usage["cached_tokens"] == 0


def test_gemini_cached_content_is_counted(capture_server) -> None:
    capture_server.set_response_sequence(
        [
            (
                200,
                {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": "ok"}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 5000,
                        "cachedContentTokenCount": 4096,
                        "candidatesTokenCount": 2,
                        "totalTokenCount": 5002,
                    },
                },
            )
        ]
    )
    client = LLMClient(
        provider="gemini", model="gemini-3.8-flash", api_key="k", base_url=capture_server.url
    )
    usage = _ask(client).usage
    assert usage["prompt_tokens"] == 5000 and usage["cached_tokens"] == 4096


def test_a_streamed_call_reports_cache_and_reasoning_counts(capture_server) -> None:
    def sse(*events: tuple[str, dict]) -> str:
        return "".join(f"event: {n}\ndata: {json.dumps(d)}\n\n" for n, d in events)

    stream = sse(
        (
            "message_start",
            {
                "message": {
                    "usage": {
                        "input_tokens": 4,
                        "cache_read_input_tokens": 4829,
                        "cache_creation_input_tokens": 0,
                    }
                }
            },
        ),
        ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "391"}}),
        ("content_block_stop", {"index": 0}),
        (
            "message_delta",
            {
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 20, "output_tokens_details": {"thinking_tokens": 17}},
            },
        ),
    )
    capture_server.set_response_sequence([(200, stream, "text/event-stream")])

    async def drain() -> list[Any]:
        client = _anthropic(capture_server.url, cache=True)
        return [e async for e in client.astream([UserMessage("q")])]

    (usage,) = [e for e in asyncio.run(drain()) if isinstance(e, Usage)]
    assert usage.prompt_tokens == 4 + 4829
    assert usage.cached_tokens == 4829 and usage.cache_write_tokens == 0
    assert usage.reasoning_tokens == 17


# ── cost: each part at its own rate (X18) ──────────────────────────────────


def test_an_openai_cache_hit_is_priced_at_the_cached_rate() -> None:
    usage = {"prompt_tokens": 3629, "completion_tokens": 10, "cached_tokens": 3456}
    usd, known = usage_cost("gpt-5-mini", usage, provider="openai")
    assert known
    assert usd == pytest.approx((173 * 0.25 + 3456 * 0.025 + 10 * 2.00) / M)


def test_a_gpt6_cache_write_is_priced_at_the_write_rate() -> None:
    usd, _ = usage_cost("gpt-6-luna", OPENAI_CACHED["usage"] | {"cache_write_tokens": 3626})
    assert usd == pytest.approx((3 * 0.10 + 3626 * 0.125 + 4 * 0.50) / M)


def test_anthropic_writes_and_reads_are_priced() -> None:
    n = STANDARD  # per n tokens, so the figures are the per-1M rates / 10
    five_min = {"prompt_tokens": n, "completion_tokens": 0, "cache_write_tokens": n}
    one_hour = five_min | {"cache_write_1h_tokens": n}
    read = {"prompt_tokens": n, "completion_tokens": 0, "cached_tokens": n}
    assert usage_cost("claude-haiku-5-5", five_min)[0] == pytest.approx(0.125 / 10)
    assert usage_cost("claude-haiku-5-5", one_hour)[0] == pytest.approx(0.20 / 10)
    assert usage_cost("claude-haiku-5-5", read)[0] == pytest.approx(0.01 / 10)
    assert usage_cost("claude-opus-5-5", read)[0] == pytest.approx(0.20 / 10)


def test_an_agent_run_costs_its_cache_tokens(capture_server) -> None:
    capture_server.set_response_sequence([(200, ANTHROPIC_CACHED)])
    agent = Agent(
        name="cached", system_prompt="rules", llm=_anthropic(capture_server.url, cache=True)
    )
    result = agent.run("q", trace=False)
    expected = (4 * 0.10 + 4829 * 0.125 + 21 * 0.50) / M
    assert result.cost == pytest.approx(expected)


def test_reasoning_tokens_are_billed_at_the_output_rate() -> None:
    """They are already inside ``completion_tokens`` (OpenAI, Anthropic, Gemini)."""
    usage = {"prompt_tokens": 10, "completion_tokens": 40, "reasoning_tokens": 32}
    assert usage_cost("gpt-5-mini", usage)[0] == pytest.approx((10 * 0.25 + 40 * 2.00) / M)


@pytest.mark.parametrize(
    ("model", "prompt", "in_rate", "out_rate"),
    [
        ("gpt-6-luna", 300_000, 0.20, 0.75),  # > 272k
        ("gpt-6-luna", 200_000, 0.10, 0.50),
        ("gpt-5.5", 300_000, 10.00, 45.00),
        ("claude-haiku-5-5", 150_000, 0.50, 2.50),  # > 100k
        ("claude-haiku-5-5", 90_000, 0.10, 0.50),
        ("gemini-3.1-pro-preview", 250_000, 4.00, 18.00),  # > 200k
    ],
)
def test_a_long_prompt_is_priced_at_the_long_context_tier(model, prompt, in_rate, out_rate) -> None:
    usd = compute_cost_usd(model, prompt, 1000)
    assert usd == pytest.approx((prompt * in_rate + 1000 * out_rate) / M)


# ── the table is current (X15) ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "inp", "out"),
    [
        ("gpt-6-astra", 10.00, 50.00),
        ("gpt-6.1-sol", 2.00, 10.00),
        ("gpt-6-sol", 2.00, 10.00),
        ("gpt-6-luna", 0.10, 0.50),
        ("gpt-5.6-terra", 2.00, 12.00),
        ("gpt-5.5", 5.00, 30.00),
        ("gpt-5.4-mini", 0.75, 4.50),
        ("gpt-5.2", 1.75, 14.00),
        ("gpt-5.1", 1.25, 10.00),
        ("gpt-5-2025-08-07", 1.25, 10.00),
        ("claude-opus-5-5", 4.00, 20.00),
        ("claude-sonnet-5-5", 2.00, 10.00),
        ("claude-sonnet-5", 2.00, 10.00),
        ("claude-haiku-5-5", 0.10, 0.50),
        ("claude-fable-5-1", 10.00, 50.00),
        ("claude-opus-4-8", 5.00, 25.00),
        ("claude-opus-4-20250514", 15.00, 75.00),
    ],
)
def test_current_list_prices(model, inp, out) -> None:
    # Below every long-context threshold (100k is the lowest, Haiku 5.5's).
    assert compute_cost_usd(model, STANDARD, 0) == pytest.approx(inp * STANDARD / M)
    assert compute_cost_usd(model, 0, M) == pytest.approx(out)


@pytest.mark.parametrize("model", ["gpt-5.7-sol", "gpt-5.9", "gemini-3.9-flash"])
def test_a_later_version_is_unpriced_not_mispriced(model) -> None:
    assert compute_cost_usd(model, M, M) is None
    assert usage_cost(model, {"prompt_tokens": 10, "completion_tokens": 5}) == (0.0, False)


def test_scheduled_increases_carry_their_cached_rate() -> None:
    cached = {"prompt_tokens": M, "completion_tokens": 0, "cached_tokens": M}
    from fastaiagent._internal import pricing

    rate = pricing._match("gemini-3.8-flash", on=date(2027, 1, 2))
    assert rate is not None and rate.cached_input_per_m == pytest.approx(0.15)
    assert usage_cost("gemini-3.8-flash", cached)[0] > 0


# ── overrides take cached and write rates ──────────────────────────────────


def test_an_override_can_set_cached_and_write_rates() -> None:
    set_rate_overrides({"gpt-6-luna": (1.0, 2.0, 0.5, 1.5)})
    usage = {
        "prompt_tokens": 300,
        "completion_tokens": 100,
        "cached_tokens": 100,
        "cache_write_tokens": 100,
    }
    assert usage_cost("gpt-6-luna", usage)[0] == pytest.approx(
        (100 * 1.0 + 100 * 0.5 + 100 * 1.5 + 100 * 2.0) / M
    )


def test_a_two_rate_override_still_works() -> None:
    set_rate_overrides({"gpt-6-luna": (1.0, 2.0)})
    usage = {"prompt_tokens": 200, "completion_tokens": 0, "cached_tokens": 100}
    assert usage_cost("gpt-6-luna", usage)[0] == pytest.approx(200 * 1.0 / M)


def test_models_json_takes_cached_rates(tmp_path, monkeypatch) -> None:
    from fastaiagent.ui import model_catalog

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    catalog = {
        "pricing": {
            "my-model": {
                "input_per_1m": 1.0,
                "output_per_1m": 2.0,
                "cached_input_per_1m": 0.1,
                "cache_write_per_1m": 1.25,
            }
        }
    }
    path = model_catalog.catalog_path(str(db))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog))
    reload_rate_overrides(str(db))
    usage = {
        "prompt_tokens": 300,
        "completion_tokens": 0,
        "cached_tokens": 100,
        "cache_write_tokens": 100,
    }
    assert usage_cost("my-model", usage)[0] == pytest.approx(
        (100 * 1.0 + 100 * 0.1 + 100 * 1.25) / M
    )


# ── Bedrock (real boto3) ───────────────────────────────────────────────────


@pytest.fixture
def _dummy_aws(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDUMMY")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "dummy-secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def _lookup(order: str) -> str:
    """Look up an order."""
    return order


def test_bedrock_cache_points_and_counts(capture_server, _dummy_aws) -> None:
    pytest.importorskip("boto3")
    capture_server.set_response_sequence(
        [
            (
                200,
                {
                    "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
                    "stopReason": "end_turn",
                    "usage": {
                        "inputTokens": 7,
                        "outputTokens": 3,
                        "cacheReadInputTokens": 1000,
                        "cacheWriteInputTokens": 24,
                        "totalTokens": 1034,
                    },
                    "metrics": {"latencyMs": 1},
                },
            )
        ]
    )
    client = LLMClient(
        provider="bedrock",
        model="us.anthropic.claude-haiku-5-5-v1:0",
        base_url=capture_server.url,
        cache=True,
    )
    tools = [FunctionTool(name="lookup", fn=_lookup).to_openai_format()]
    response = _ask(client, [SystemMessage("rules"), UserMessage("q")], tools=tools)
    body = capture_server.requests[0]["body"]
    point = {"cachePoint": {"type": "default"}}
    assert body["system"][-1] == point
    assert body["toolConfig"]["tools"][-1] == point
    assert body["messages"][-1]["content"][-1] == point
    assert response.usage["prompt_tokens"] == 7 + 1000 + 24
    assert response.usage["cached_tokens"] == 1000 and response.usage["cache_write_tokens"] == 24


# ── serialization (signed off 2026-10-18: reasoning and cache are in to_dict) ──


def test_reasoning_and_cache_round_trip_through_to_dict() -> None:
    client = LLMClient(provider="anthropic", model="claude-haiku-5-5", reasoning="low", cache="1h")
    data = client.to_dict()
    assert data["reasoning"] == "low" and data["cache"] == "1h"
    back = LLMClient.from_dict(data)
    assert back.reasoning == "low" and back.cache == "1h"


def test_a_default_client_payload_is_unchanged() -> None:
    data = LLMClient(provider="openai", model="gpt-5-mini").to_dict()
    assert data == {"provider": "openai", "model": "gpt-5-mini"}


# ── spans: counts always, reasoning text only with payloads on ─────────────


def test_spans_carry_the_counts_and_gate_the_reasoning_text(
    capture_server, monkeypatch, tmp_path
) -> None:
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    reset_config()
    otel.reset()
    try:
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")
        local, egress = InMemorySpanExporter(), InMemorySpanExporter()
        otel.add_span_processor(SimpleSpanProcessor(local))
        otel.add_exporter(egress)
        body = json.loads(json.dumps(ANTHROPIC_CACHED))
        body["content"].insert(0, {"type": "thinking", "thinking": "secret plan", "signature": "s"})
        body["usage"]["output_tokens_details"] = {"thinking_tokens": 15}
        capture_server.set_response_sequence([(200, body)])
        _ask(_anthropic(capture_server.url, cache=True, reasoning="low"))
        otel.get_tracer_provider().force_flush()

        def llm_attrs(exporter: InMemorySpanExporter) -> dict[str, Any]:
            (span,) = [s for s in exporter.get_finished_spans() if s.name.startswith("llm.")]
            return dict(span.attributes)

        here, out = llm_attrs(local), llm_attrs(egress)
        for attrs in (here, out):
            assert attrs["gen_ai.usage.input_tokens"] == 4 + 4829
            assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 4829
            assert attrs["gen_ai.usage.cache_read.input_tokens"] == 0
            assert attrs["gen_ai.usage.reasoning_tokens"] == 15
        assert here["gen_ai.response.reasoning"] == "secret plan"
        assert "gen_ai.response.reasoning" not in out
        assert "secret plan" not in json.dumps(out, default=str)
    finally:
        otel.reset()
        reset_config()
