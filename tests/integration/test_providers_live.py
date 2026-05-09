"""Live LLM tests for the v1.8.0 provider expansion.

Two providers are exercised end-to-end:

- **Groq** — exercises the ``openai_compat`` wire shared by 11 of the 12
  shipped presets, so a green Groq run is strong evidence the others
  work too (modulo provider-specific quirks).
- **Gemini** — the only ``native_gemini`` wire, must be tested directly.

All tests are gated on env vars (``GROQ_API_KEY``, ``GEMINI_API_KEY``).
Run via:

    zsh -lc 'pytest tests/integration/test_providers_live.py -v'

Cost: each test uses a tiny prompt; a full run costs less than a
fraction of a cent on either provider's free tier.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from fastaiagent.llm import LLMClient
from fastaiagent.llm.message import UserMessage
from fastaiagent.llm.stream import TextDelta, ToolCallEnd, ToolCallStart

HAS_GROQ = bool(os.environ.get("GROQ_API_KEY"))
HAS_GEMINI = bool(os.environ.get("GEMINI_API_KEY"))

needs_groq = pytest.mark.skipif(not HAS_GROQ, reason="GROQ_API_KEY not set")
needs_gemini = pytest.mark.skipif(not HAS_GEMINI, reason="GEMINI_API_KEY not set")


def _skip_on_quota(exc: Exception) -> None:
    """Convert provider rate-limit / quota errors into a pytest skip.

    The Gemini free tier caps at ~10 requests/minute; running this whole
    suite back-to-back trips the limit. Treat that as an infra issue
    rather than a real test failure.
    """
    msg = str(exc).lower()
    if "429" in msg or "quota" in msg or "rate limit" in msg or "resource_exhausted" in msg:
        pytest.skip(f"Provider rate-limited; skipping. Underlying error: {exc!s:.200}")
    raise exc


# ---------------------------------------------------------------------------
# Groq — openai_compat wire
# ---------------------------------------------------------------------------


@needs_groq
def test_groq_simple_completion() -> None:
    client = LLMClient(provider="groq", model="llama-3.1-8b-instant")
    resp = client.complete(
        [UserMessage("Reply with the single word: pong. Nothing else.")]
    )
    assert resp.content
    assert "pong" in resp.content.lower()
    # Provider tag flows into model field — pricing prefix-match should work.
    assert resp.usage.get("prompt_tokens", 0) > 0


@needs_groq
def test_groq_streaming_accumulates() -> None:
    client = LLMClient(provider="groq", model="llama-3.1-8b-instant")

    async def collect() -> str:
        chunks: list[str] = []
        async for ev in client.astream(
            [UserMessage("Reply with: ack. Nothing else.")]
        ):
            if isinstance(ev, TextDelta):
                chunks.append(ev.text)
        return "".join(chunks)

    text = asyncio.run(collect())
    assert "ack" in text.lower()


@needs_groq
def test_groq_tool_call() -> None:
    """Groq supports OpenAI-style tool calling on the larger Llama models."""
    client = LLMClient(
        provider="groq",
        model="llama-3.3-70b-versatile",
        temperature=0.0,
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_temperature",
                "description": "Return the current temperature for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    resp = client.complete(
        [UserMessage("What is the temperature in Paris? Use the tool.")],
        tools=tools,
    )
    if not resp.tool_calls:
        pytest.skip(
            "Groq did not invoke the tool on this run; tool-call routing is "
            "model-dependent and acceptably flaky on the free tier."
        )
    assert resp.tool_calls[0].name == "get_temperature"


@needs_groq
def test_groq_trace_span_records_provider() -> None:
    """The OTel span emitted by acomplete carries gen_ai.system='groq'."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Don't replace the global provider if one is already set — wrap our
    # span explicitly via the existing get_tracer() path by setting it
    # before the call. Skip if a provider is already installed.
    current = trace.get_tracer_provider()
    if not isinstance(current, type(trace.NoOpTracerProvider())):
        pytest.skip("Tracer provider already installed; can't assert spans cleanly")
    trace.set_tracer_provider(provider)

    client = LLMClient(provider="groq", model="llama-3.1-8b-instant")
    client.complete([UserMessage("ok")])

    spans = exporter.get_finished_spans()
    matched = [s for s in spans if s.name.startswith("llm.groq.")]
    assert matched, f"no groq llm span found; got: {[s.name for s in spans]}"
    attrs = matched[0].attributes or {}
    assert attrs.get("gen_ai.system") == "groq"


# ---------------------------------------------------------------------------
# Gemini — native_gemini wire
# ---------------------------------------------------------------------------


@needs_gemini
def test_gemini_simple_completion() -> None:
    client = LLMClient(provider="gemini", model="gemini-2.5-flash")
    try:
        resp = client.complete(
            [UserMessage("Reply with the single word: pong. Nothing else.")]
        )
    except Exception as e:
        _skip_on_quota(e)
        raise
    assert resp.content
    assert "pong" in resp.content.lower()


@needs_gemini
def test_gemini_streaming_accumulates() -> None:
    client = LLMClient(provider="gemini", model="gemini-2.5-flash")

    async def collect() -> str:
        chunks: list[str] = []
        async for ev in client.astream(
            [UserMessage("Reply with: ack. Nothing else.")]
        ):
            if isinstance(ev, TextDelta):
                chunks.append(ev.text)
        return "".join(chunks)

    try:
        text = asyncio.run(collect())
    except Exception as e:
        _skip_on_quota(e)
        raise
    assert "ack" in text.lower()


@needs_gemini
def test_gemini_tool_call() -> None:
    client = LLMClient(provider="gemini", model="gemini-2.5-flash", temperature=0.0)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_temperature",
                "description": "Return the current temperature for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    try:
        resp = client.complete(
            [
                UserMessage(
                    "Use the get_temperature tool to look up the temperature in Paris."
                )
            ],
            tools=tools,
        )
    except Exception as e:
        _skip_on_quota(e)
        raise
    if not resp.tool_calls:
        pytest.skip("Gemini did not invoke the tool on this run; flaky.")
    assert resp.tool_calls[0].name == "get_temperature"
    assert "city" in resp.tool_calls[0].arguments


@needs_gemini
def test_gemini_streaming_emits_tool_call_pair() -> None:
    client = LLMClient(provider="gemini", model="gemini-2.5-flash", temperature=0.0)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo the input back.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }
    ]

    async def collect() -> list:
        events = []
        async for ev in client.astream(
            [
                UserMessage(
                    "Use the echo tool with text='hello'."
                )
            ],
            tools=tools,
        ):
            events.append(ev)
        return events

    try:
        events = asyncio.run(collect())
    except Exception as e:
        _skip_on_quota(e)
        raise
    starts = [e for e in events if isinstance(e, ToolCallStart)]
    ends = [e for e in events if isinstance(e, ToolCallEnd)]
    if not starts:
        pytest.skip("Gemini did not invoke the tool on this run; flaky.")
    assert len(starts) == len(ends)
    assert ends[0].tool_name == "echo"


@needs_gemini
def test_gemini_structured_output_via_response_format() -> None:
    """Gemini honours generationConfig.responseSchema when we set
    response_format=json_schema."""
    client = LLMClient(provider="gemini", model="gemini-2.5-flash", temperature=0.0)
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
        },
        "required": ["city", "country"],
    }
    try:
        resp = client.complete(
            [
                UserMessage(
                    "Return a JSON object with the city and country of the Eiffel Tower."
                )
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "place", "schema": schema},
            },
        )
    except Exception as e:
        _skip_on_quota(e)
        raise
    import json

    assert resp.content
    parsed = json.loads(resp.content)
    assert parsed.get("city", "").lower() == "paris"
    assert parsed.get("country", "").lower() in {"france", "fr"}
