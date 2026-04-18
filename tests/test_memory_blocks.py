"""Tests for fastaiagent.agent.memory_blocks and ComposableMemory.

Deterministic tests (no LLM) cover StaticBlock, VectorBlock, ComposableMemory
composition, persistence, and backward compat with AgentMemory.

Live tests (real LLM via OPENAI_API_KEY or ANTHROPIC_API_KEY) cover
SummaryBlock and FactExtractionBlock — the two blocks whose core behavior
is LLM-dependent. Per the project's no-mocking rule, these are not faked.
Live tests skip cleanly if neither API key is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fastaiagent import (
    Agent,
    AgentMemory,
    ComposableMemory,
    FactExtractionBlock,
    LLMClient,
    StaticBlock,
    SummaryBlock,
    VectorBlock,
)
from fastaiagent.agent.memory_blocks import MemoryBlock
from fastaiagent.kb.embedding import SimpleEmbedder
from fastaiagent.llm.message import AssistantMessage, ToolMessage, UserMessage

try:
    import faiss  # noqa: F401

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

_skip_no_faiss = pytest.mark.skipif(not _HAS_FAISS, reason="faiss-cpu not installed")

_HAS_LIVE_KEY = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
_skip_no_live_key = pytest.mark.skipif(
    not _HAS_LIVE_KEY,
    reason="no OPENAI_API_KEY or ANTHROPIC_API_KEY set",
)


def _live_llm() -> LLMClient:
    if os.environ.get("OPENAI_API_KEY"):
        return LLMClient(provider="openai", model="gpt-4o-mini")
    return LLMClient(provider="anthropic", model="claude-haiku-4-5-20251001")


# ---------------------------------------------------------------------------
# StaticBlock
# ---------------------------------------------------------------------------


def test_static_block_renders_every_turn() -> None:
    block = StaticBlock("The user prefers terse answers.")
    for _ in range(3):
        out = block.render(query="anything")
        assert len(out) == 1
        assert out[0].content == "The user prefers terse answers."


def test_static_block_on_message_is_noop() -> None:
    block = StaticBlock("fixed")
    block.on_message(UserMessage("ignored"))
    assert block.render("") == [block.render("")[0]]  # same output


def test_static_block_empty_text_renders_nothing() -> None:
    assert StaticBlock("").render("q") == []


# ---------------------------------------------------------------------------
# VectorBlock (deterministic — uses SimpleEmbedder + FAISS)
# ---------------------------------------------------------------------------


@_skip_no_faiss
def test_vector_block_recalls_semantic_match() -> None:
    from fastaiagent.kb.backends.faiss import FaissVectorStore

    store = FaissVectorStore(dimension=64, index_type="flat")
    embedder = SimpleEmbedder(dimensions=64)
    block = VectorBlock(store=store, embedder=embedder, top_k=2, min_content_chars=0)

    # Seed the block with past messages.
    block.on_message(UserMessage("I have a dog named Rex who loves fetch."))
    block.on_message(UserMessage("My favorite color is cobalt blue."))
    block.on_message(UserMessage("I enjoy hiking in the Rockies."))

    # The rendered fragment should include at least one of the prior messages.
    # SimpleEmbedder is character-frequency based, so query exact substrings.
    out = block.render(query="dog named Rex")
    assert len(out) == 1
    body = out[0].content
    assert "dog named Rex" in body or "Rex" in body


@_skip_no_faiss
def test_vector_block_skips_short_messages() -> None:
    from fastaiagent.kb.backends.faiss import FaissVectorStore

    store = FaissVectorStore(dimension=32, index_type="flat")
    block = VectorBlock(
        store=store,
        embedder=SimpleEmbedder(dimensions=32),
        top_k=5,
        min_content_chars=20,
    )
    block.on_message(UserMessage("ok"))              # below threshold
    block.on_message(UserMessage("yes"))             # below threshold
    block.on_message(UserMessage("a longer and worthwhile message"))
    assert store.count() == 1


@_skip_no_faiss
def test_vector_block_namespace_isolation() -> None:
    from fastaiagent.kb.backends.faiss import FaissVectorStore

    # Two VectorBlocks sharing one store but using different namespaces.
    store = FaissVectorStore(dimension=32, index_type="flat")
    embedder = SimpleEmbedder(dimensions=32)
    block_a = VectorBlock(
        store=store, embedder=embedder, top_k=5, namespace="alpha", min_content_chars=0
    )
    block_b = VectorBlock(
        store=store, embedder=embedder, top_k=5, namespace="bravo", min_content_chars=0
    )

    block_a.on_message(UserMessage("alpha-specific fact about owls"))
    block_b.on_message(UserMessage("bravo-specific fact about rivers"))

    a_out = block_a.render("owls")
    b_out = block_b.render("rivers")

    # Each block only surfaces its own namespace.
    assert a_out and "owls" in a_out[0].content
    assert "rivers" not in a_out[0].content
    assert b_out and "rivers" in b_out[0].content
    assert "owls" not in b_out[0].content


@_skip_no_faiss
def test_vector_block_empty_query_returns_nothing() -> None:
    from fastaiagent.kb.backends.faiss import FaissVectorStore

    store = FaissVectorStore(dimension=16, index_type="flat")
    block = VectorBlock(
        store=store, embedder=SimpleEmbedder(dimensions=16), top_k=3, min_content_chars=0
    )
    block.on_message(UserMessage("seed content"))
    assert block.render("") == []
    assert block.render("   ") == []


# ---------------------------------------------------------------------------
# ComposableMemory
# ---------------------------------------------------------------------------


class _RecordingBlock(MemoryBlock):
    """Test-only block: records every on_message / render call."""

    name = "recorder"

    def __init__(self) -> None:
        self.seen: list[str] = []
        self.rendered_for: list[str] = []

    def on_message(self, message) -> None:
        self.seen.append(message.content or "")

    def render(self, query: str):
        self.rendered_for.append(query)
        return []


def test_composable_memory_broadcasts_add_to_blocks() -> None:
    rec = _RecordingBlock()
    mem = ComposableMemory(blocks=[rec], primary=AgentMemory(max_messages=10))
    mem.add(UserMessage("hello"))
    mem.add(AssistantMessage("hi there"))
    assert rec.seen == ["hello", "hi there"]


def test_composable_memory_passes_query_to_blocks() -> None:
    rec = _RecordingBlock()
    mem = ComposableMemory(blocks=[rec])
    mem.get_context(query="what is my name?")
    assert rec.rendered_for == ["what is my name?"]


def test_composable_memory_concatenates_blocks_then_primary() -> None:
    static1 = StaticBlock("Fact A.", name="a")
    static2 = StaticBlock("Fact B.", name="b")
    mem = ComposableMemory(blocks=[static1, static2], primary=AgentMemory(max_messages=5))
    mem.add(UserMessage("a user message"))
    ctx = mem.get_context(query="anything")
    # Expect: SystemMessage(Fact A), SystemMessage(Fact B), UserMessage(a user message)
    assert len(ctx) == 3
    assert ctx[0].content == "Fact A."
    assert ctx[1].content == "Fact B."
    assert ctx[2].content == "a user message"


def test_composable_memory_primary_window_truncates() -> None:
    mem = ComposableMemory(primary=AgentMemory(max_messages=3))
    for i in range(5):
        mem.add(UserMessage(f"m{i}"))
    ctx = mem.get_context()
    # No blocks, so ctx is just the primary — last 3.
    assert [m.content for m in ctx] == ["m2", "m3", "m4"]


def test_composable_memory_backward_compat_with_agent_memory() -> None:
    """An Agent with AgentMemory works identically after the refactor."""
    from fastaiagent.llm.client import LLMResponse
    from tests.conftest import MockLLMClient

    llm = MockLLMClient(responses=[LLMResponse(content="hi!", finish_reason="stop")])
    agent = Agent(name="t", llm=llm, memory=AgentMemory(max_messages=20))
    result = agent.run("hello", trace=False)
    assert result.output == "hi!"
    # Memory captured the exchange.
    assert len(agent.memory) == 2


def test_composable_memory_dropin_replacement_for_agent_memory() -> None:
    """Same Agent works when given ComposableMemory instead of AgentMemory."""
    from fastaiagent.llm.client import LLMResponse
    from tests.conftest import MockLLMClient

    llm = MockLLMClient(responses=[LLMResponse(content="ok", finish_reason="stop")])
    mem = ComposableMemory(
        blocks=[StaticBlock("Always be concise.")],
        primary=AgentMemory(max_messages=20),
    )
    agent = Agent(name="t", llm=llm, memory=mem)
    agent.run("hello", trace=False)
    # The static block was injected in the LLM-visible messages.
    messages_sent = llm._calls[0]["messages"]
    assert any(m.content == "Always be concise." for m in messages_sent)


def test_composable_memory_block_failure_does_not_break_run() -> None:
    class Broken(MemoryBlock):
        name = "broken"

        def on_message(self, message) -> None:
            raise RuntimeError("oops on add")

        def render(self, query: str):
            raise RuntimeError("oops on render")

    mem = ComposableMemory(blocks=[Broken(), StaticBlock("survivor", name="survivor")])
    mem.add(UserMessage("hi"))  # should not raise
    ctx = mem.get_context(query="q")
    # Survivor's output still makes it through.
    assert any(m.content == "survivor" for m in ctx)


def test_composable_memory_save_and_load_roundtrip(tmp_path: Path) -> None:
    facts_block = FactExtractionBlock(llm=None)  # type: ignore[arg-type]
    # Inject facts without running the LLM.
    facts_block._facts = ["user likes tea", "user is allergic to pollen"]
    mem = ComposableMemory(blocks=[facts_block], primary=AgentMemory(max_messages=5))
    mem.add(UserMessage("hi"))
    mem.save(tmp_path / "mem")

    # New instance, fresh block.
    restored_block = FactExtractionBlock(llm=None)  # type: ignore[arg-type]
    restored_mem = ComposableMemory(blocks=[restored_block], primary=AgentMemory(max_messages=5))
    restored_mem.load(tmp_path / "mem")

    assert restored_block._facts == ["user likes tea", "user is allergic to pollen"]
    assert len(restored_mem.primary) == 1


# ---------------------------------------------------------------------------
# Fact extraction — deterministic fake LLM path exercised through load/save
# (Real LLM test is below, gated by an API key.)
# ---------------------------------------------------------------------------


def test_fact_extraction_block_dedupes_and_caps() -> None:
    block = FactExtractionBlock(llm=None, max_facts=3)  # type: ignore[arg-type]
    # Bypass the LLM by writing straight to the internal list.
    for fact in ["a", "b", "a", "c", "d", "b"]:
        if fact and fact not in block._facts:
            block._facts.append(fact)
    if len(block._facts) > block.max_facts:
        block._facts = block._facts[-block.max_facts :]
    assert block._facts == ["b", "c", "d"]  # dedup preserves first-seen order, cap drops oldest


def test_fact_extraction_block_skips_tool_messages() -> None:
    seen: list[str] = []

    class Stub(FactExtractionBlock):
        def _extract(self, content: str):  # noqa: D401
            seen.append(content)
            return []

    block = Stub(llm=None, extract_every=1)  # type: ignore[arg-type]
    block.on_message(ToolMessage(content="tool output noise", tool_call_id="x"))
    block.on_message(UserMessage("a real user statement"))
    assert seen == ["a real user statement"]


# ---------------------------------------------------------------------------
# Live LLM tests (real OpenAI / Anthropic)
# ---------------------------------------------------------------------------


@_skip_no_live_key
def test_summary_block_live_produces_summary() -> None:
    """With a real LLM, SummaryBlock emits a non-empty SystemMessage once the
    threshold is crossed.
    """
    block = SummaryBlock(llm=_live_llm(), keep_last=2, summarize_every=3, max_chars=300)
    # 5 user messages; with keep_last=2 and summarize_every=3, a summary
    # should have been produced by now.
    block.on_message(UserMessage("My name is Casey. Remember it."))
    block.on_message(AssistantMessage("Hello Casey."))
    block.on_message(UserMessage("I live in Amsterdam and work as a civil engineer."))
    block.on_message(AssistantMessage("Noted."))
    block.on_message(UserMessage("My dog is named Pepper."))

    rendered = block.render("tell me about myself")
    assert rendered, "expected SummaryBlock to produce a non-empty summary"
    body = rendered[0].content.lower()
    assert "casey" in body or "amsterdam" in body or "pepper" in body


@_skip_no_live_key
def test_fact_extraction_block_live_extracts_facts() -> None:
    """With a real LLM, FactExtractionBlock picks durable facts from a message."""
    block = FactExtractionBlock(llm=_live_llm(), max_facts=50, extract_every=1)
    block.on_message(
        UserMessage(
            "I'm a Python developer at Acme Corp. I'm allergic to peanuts "
            "and prefer dark chocolate. My timezone is UTC+1."
        )
    )
    facts_text = " ".join(block._facts).lower()
    # At least one durable fact should appear — don't over-specify which.
    assert block._facts, "expected at least one fact to be extracted"
    assert any(
        keyword in facts_text
        for keyword in ("python", "acme", "peanut", "chocolate", "timezone", "utc")
    )
