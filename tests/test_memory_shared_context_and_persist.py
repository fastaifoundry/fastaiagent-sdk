"""Shared memory context (Feature 3) + persist-during-run (Feature 2).

Deterministic — no LLM, no mocking of the SDK. Uses the same fake store/embedder
the scoring tests use. The persist path is exercised against a real SQLite
``MemoryStore``; the LLM-driven extraction path is covered by the e2e gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastaiagent._internal.config import reset_config
from fastaiagent.agent.memory import ComposableMemory
from fastaiagent.agent.memory_blocks import (
    FactExtractionBlock,
    MemoryBlock,
    MemoryIsolationError,
    SharedMemoryContext,
    StaticBlock,
    VectorBlock,
)
from fastaiagent.llm.message import SystemMessage


class _FakeEmbedder:
    def embed(self, texts):
        return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]


class _CannedStore:
    def __init__(self, hits):
        self._hits = hits

    def add(self, chunks, embeddings):
        pass

    def search(self, embedding, top_k):
        return list(self._hits)[:top_k]


def _chunk(content: str):
    from fastaiagent.kb.chunking import Chunk

    return Chunk(
        id=content,
        content=content,
        metadata={"namespace": "default"},
        index=0,
        start_char=0,
        end_char=len(content),
    )


# ---------------------------------------------------------------------------
# Feature 3 — shared memory context
# ---------------------------------------------------------------------------


def test_render_with_context_default_delegates_to_render():
    """A block that only implements render() still works via the new path."""

    class Custom(MemoryBlock):
        name = "custom"

        def on_message(self, message):
            return

        def render(self, query):
            return [SystemMessage("hello")]

    block = Custom()
    shared = SharedMemoryContext(query="q")
    assert block.render_with_context("q", shared) == block.render("q")


def test_later_block_sees_upstream_output():
    """SharedMemoryContext carries earlier blocks' output to later blocks."""
    seen = {}

    class Downstream(MemoryBlock):
        name = "down"

        def on_message(self, message):
            return

        def render(self, query):
            return []

        def render_with_context(self, query, shared):
            seen["upstream"] = shared.upstream_text()
            seen["by_static"] = shared.by_block("static")
            return []

    mem = ComposableMemory(blocks=[StaticBlock("PINNED FACT"), Downstream()])
    mem.get_context(query="q")
    assert "PINNED FACT" in seen["upstream"]
    assert any("PINNED FACT" in (m.content or "") for m in seen["by_static"])


def test_vectorblock_dedupes_against_upstream():
    # StaticBlock upstream literally contains the recalled line → it's dropped.
    store = _CannedStore(
        [
            (_chunk("[user] I adopted a beagle named Biscuit"), 0.9),
            (_chunk("[user] I like hiking"), 0.7),
        ]
    )
    mem = ComposableMemory(
        blocks=[
            StaticBlock("Known: I adopted a beagle named Biscuit. Lives in Seattle."),
            VectorBlock(
                store=store,
                embedder=_FakeEmbedder(),
                top_k=5,
                dedupe_against_upstream=True,
            ),
        ]
    )
    out = mem.get_context(query="q")
    report = mem.blocks[1].last_render_report()
    assert report.deduped_count == 1
    assert report.rendered_count == 1
    joined = " ".join(m.content or "" for m in out)
    assert "I like hiking" in joined
    assert "beagle named Biscuit" in joined  # from StaticBlock, not re-recalled by vector
    # Called standalone with empty upstream → no dedup → both come back.
    mem.blocks[1].render_with_context("q", SharedMemoryContext(query="q"))
    assert mem.blocks[1].last_render_report().deduped_count == 0


def test_vectorblock_no_dedupe_by_default():
    store = _CannedStore([(_chunk("[user] I adopted a beagle named Biscuit"), 0.9)])
    mem = ComposableMemory(
        blocks=[
            StaticBlock("Known: I adopted a beagle named Biscuit."),
            VectorBlock(store=store, embedder=_FakeEmbedder(), top_k=5),  # flag off
        ]
    )
    mem.get_context(query="q")
    report = mem.blocks[1].last_render_report()
    assert report.deduped_count is None  # dedup not active
    assert report.rendered_count == 1


# ---------------------------------------------------------------------------
# Feature 2 — persist-during-run
# ---------------------------------------------------------------------------


@pytest.fixture
def _db(monkeypatch, tmp_path: Path):
    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    reset_config()
    yield db
    reset_config()


def test_persist_requires_scope_id():
    with pytest.raises(ValueError, match="scope_id is required"):
        FactExtractionBlock(llm=None, persist=True, scope="user")


def test_persist_rejects_bad_scope():
    with pytest.raises(ValueError, match="scope must be one of"):
        FactExtractionBlock(llm=None, persist=True, scope="bogus", scope_id="x")


def test_isolated_copy_raises_when_persist():
    block = FactExtractionBlock(llm=None, persist=True, scope="user", scope_id="u")
    with pytest.raises(MemoryIsolationError):
        block.isolated_copy()


def test_isolated_copy_ok_when_not_persist():
    block = FactExtractionBlock(llm=None, persist=False)
    copy = block.isolated_copy()
    assert isinstance(copy, FactExtractionBlock)
    assert copy.persist is False


def test_persist_facts_writes_to_store_with_confidence(_db):
    from fastaiagent.learn import MemoryStore

    block = FactExtractionBlock(
        llm=None, persist=True, scope="user", scope_id="upendra", confidence=0.6
    )
    written = block._persist_facts(["User has a beagle named Biscuit."])
    assert written == 1

    rows = MemoryStore(db_path=str(_db)).list_active(scope="user", scope_id="upendra")
    facts = {r.fact: r for r in rows}
    assert "User has a beagle named Biscuit." in facts
    assert facts["User has a beagle named Biscuit."].confidence == 0.6


def test_persist_facts_stamps_source_trace_id(_db):
    from fastaiagent.learn import MemoryStore
    from fastaiagent.trace.otel import get_tracer

    block = FactExtractionBlock(llm=None, persist=True, scope="user", scope_id="u")
    tracer = get_tracer("fastaiagent")
    with tracer.start_as_current_span("agent.test"):
        block._persist_facts(["A durable fact."])

    rows = MemoryStore(db_path=str(_db)).list_active(scope="user", scope_id="u")
    assert rows, "no facts persisted"
    fact = next(r for r in rows if r.fact == "A durable fact.")
    assert fact.source_trace_id  # a real hex trace id was stamped
    assert len(fact.source_trace_id) == 32


def test_persist_facts_is_idempotent(_db):
    from fastaiagent.learn import MemoryStore

    block = FactExtractionBlock(llm=None, persist=True, scope="user", scope_id="u")
    block._persist_facts(["Same fact."])
    block._persist_facts(["Same fact."])
    rows = MemoryStore(db_path=str(_db)).list_active(scope="user", scope_id="u")
    assert sum(1 for r in rows if r.fact == "Same fact.") == 1
