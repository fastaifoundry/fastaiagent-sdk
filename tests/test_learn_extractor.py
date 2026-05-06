"""Unit tests for fastaiagent.learn.extractor — gated on OPENAI_API_KEY.

The extractor's whole point is making a real LLM call, so the meaningful
tests need a key. The trace-summarization helper is pure Python and is
tested unconditionally below.
"""

from __future__ import annotations

import os

import pytest

from fastaiagent.learn import Fact, MemoryStore, extract_and_store, extract_facts_from_trace
from fastaiagent.learn.extractor import _summarize_trace_for_extraction
from fastaiagent.trace.storage import SpanData, TraceData

# ─── Pure-Python helper (always run) ─────────────────────────────────────────


def test_summarize_trace_pulls_genai_text() -> None:
    trace = TraceData(
        trace_id="t1",
        spans=[
            SpanData(
                span_id="s1",
                trace_id="t1",
                name="llm.openai.gpt-4o",
                attributes={
                    "gen_ai.request.messages": "user prompt text",
                    "gen_ai.response.content": "assistant response text",
                },
            ),
        ],
    )
    text = _summarize_trace_for_extraction(trace)
    assert "user prompt text" in text
    assert "assistant response text" in text
    assert "llm.openai.gpt-4o" in text


def test_summarize_trace_pulls_research_payloads() -> None:
    trace = TraceData(
        trace_id="t1",
        spans=[
            SpanData(
                span_id="s1",
                trace_id="t1",
                name="deep_research.research",
                attributes={
                    "fastaiagent.research.subtopic": "RAG basics",
                    "fastaiagent.research.findings": '{"summary": "lots of detail"}',
                },
            ),
        ],
    )
    text = _summarize_trace_for_extraction(trace)
    assert "deep_research.research" in text
    assert "lots of detail" in text


def test_summarize_trace_truncates_at_max_chars() -> None:
    huge = "X" * 50_000
    trace = TraceData(
        trace_id="t",
        spans=[
            SpanData(
                span_id=f"s{i}",
                trace_id="t",
                name=f"span-{i}",
                attributes={"gen_ai.response.content": huge},
            )
            for i in range(5)
        ],
    )
    text = _summarize_trace_for_extraction(trace, max_chars=2000)
    assert len(text) <= 3000  # 2000 + truncated marker overhead
    assert "[...truncated]" in text


# ─── Real LLM-driven extraction (gated) ─────────────────────────────────────


_NEEDS_LLM = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY required for real extraction call",
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    from fastaiagent._internal.config import reset_config

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    reset_config()
    yield MemoryStore()
    reset_config()


@_NEEDS_LLM
def test_extract_facts_from_trace_returns_list_of_facts(store) -> None:
    import fastaiagent as fa

    trace = TraceData(
        trace_id="t1",
        spans=[
            SpanData(
                span_id="s1",
                trace_id="t1",
                name="llm.openai.gpt-4o",
                attributes={
                    "gen_ai.request.messages": (
                        "User: I always want my reports under 500 words and in "
                        "British English. Please remember this."
                    ),
                    "gen_ai.response.content": "Got it — under 500 words, British English.",
                },
            ),
        ],
    )
    llm = fa.LLMClient(provider="openai", model="gpt-4o-mini")
    facts = extract_facts_from_trace(
        trace, llm=llm, scope="user", scope_id="test-user", max_facts=5
    )
    assert isinstance(facts, list)
    # The model should pull out the preference statement.
    assert any("500" in f.fact or "British" in f.fact or "word" in f.fact.lower() for f in facts), (
        f"expected preference fact, got {[f.fact for f in facts]!r}"
    )
    assert all(isinstance(f, Fact) for f in facts)
    assert all(f.source_trace_id == "t1" for f in facts)


@_NEEDS_LLM
def test_extract_and_store_writes_to_db(store) -> None:
    import fastaiagent as fa

    trace = TraceData(
        trace_id="t2",
        spans=[
            SpanData(
                span_id="s1",
                trace_id="t2",
                name="llm.openai.gpt-4o",
                attributes={
                    "gen_ai.request.messages": (
                        "Always cite sources for any factual claim about token costs."
                    ),
                    "gen_ai.response.content": "Will do.",
                },
            ),
        ],
    )
    llm = fa.LLMClient(provider="openai", model="gpt-4o-mini")
    result = extract_and_store(
        trace,
        llm=llm,
        store=store,
        scope="agent",
        scope_id="test-agent",
        max_facts=5,
    )
    assert result.trace_id == "t2"
    assert len(result.candidates) >= 0  # extractor may legitimately return 0
    if result.candidates:
        assert len(result.written_ids) == len(result.candidates)
        # Re-read from store to confirm round-trip.
        active = store.list_active(scope="agent", scope_id="test-agent")
        assert len(active) >= 1


@_NEEDS_LLM
def test_extract_and_store_dry_run_writes_nothing(store) -> None:
    import fastaiagent as fa

    trace = TraceData(
        trace_id="t3",
        spans=[
            SpanData(
                span_id="s1",
                trace_id="t3",
                name="llm.openai.gpt-4o",
                attributes={
                    "gen_ai.request.messages": "User prefers terse answers.",
                    "gen_ai.response.content": "OK.",
                },
            ),
        ],
    )
    llm = fa.LLMClient(provider="openai", model="gpt-4o-mini")
    result = extract_and_store(
        trace,
        llm=llm,
        store=store,
        scope="agent",
        scope_id="test-agent",
        dry_run=True,
    )
    assert result.written_ids == []
    assert store.list_active(scope="agent", scope_id="test-agent") == []
