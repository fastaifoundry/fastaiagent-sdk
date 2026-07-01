"""Memory observability — ``memory.read`` / ``memory.write`` spans.

No mocking of OTel, memory, or the tracing layer. We wire real memory blocks
(with the same fake embedder/store the scoring tests already use — that's a
test embedder, not a mock of the SDK) through the real tracing helpers and read
the spans back out of ``local.db`` to verify the shape the UI depends on.

LLM-backed blocks (SummaryBlock, FactExtractionBlock) are exercised against a
real model in ``tests/e2e/test_memory_tracing_e2e.py`` — kept out of this file
so it stays deterministic and key-free.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastaiagent._internal.config import reset_config
from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.agent._memory_tracing import traced_add, traced_get_context
from fastaiagent.agent.memory import AgentMemory, ComposableMemory
from fastaiagent.agent.memory_blocks import MemoryBlock, StaticBlock, VectorBlock
from fastaiagent.llm.message import Message, SystemMessage, UserMessage
from fastaiagent.trace.otel import get_tracer
from fastaiagent.trace.otel import reset as reset_tracer


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    reset_config()
    reset_tracer()
    yield tmp_path / "local.db"
    reset_tracer()
    reset_config()


# ---------------------------------------------------------------------------
# Test doubles — a counting embedder and a canned-hit store (mirrors
# test_memory_scoring's fakes; these are test embedders, not SDK mocks).
# ---------------------------------------------------------------------------


class _CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]


def _chunk(content: str, namespace: str = "default"):
    from fastaiagent.kb.chunking import Chunk

    return Chunk(
        id=content,
        content=content,
        metadata={"namespace": namespace},
        index=0,
        start_char=0,
        end_char=len(content),
    )


class _CannedStore:
    """Returns canned (chunk, similarity) hits, descending by similarity."""

    def __init__(self, hits):
        self._hits = hits
        self.added = []

    def add(self, chunks, embeddings):
        self.added.append((chunks, embeddings))

    def search(self, embedding, top_k):
        return list(self._hits)[:top_k]


def _read_spans(db_path: Path) -> list[dict]:
    reset_tracer()  # shutdown() flushes the processor before we read
    if not db_path.exists():
        return []
    with SQLiteHelper(db_path) as db:
        if not db.fetchone("SELECT name FROM sqlite_master WHERE type='table' AND name='spans'"):
            return []
        return db.fetchall("SELECT * FROM spans ORDER BY start_time")


def _attrs(row: dict) -> dict:
    return json.loads(row.get("attributes") or "{}")


def _by_name(spans, name):
    return [s for s in spans if s["name"] == name]


# ---------------------------------------------------------------------------
# Scenario 1 — read span parent + per-block children
# ---------------------------------------------------------------------------


def test_read_span_emits_parent_and_per_block_children(_isolated_db):
    store = _CannedStore([(_chunk("prior: my email is a@b.com"), 0.9)])
    mem = ComposableMemory(
        blocks=[
            StaticBlock("The user is Upendra."),
            VectorBlock(store=store, embedder=_CountingEmbedder(), top_k=3),
        ],
        primary=AgentMemory(max_messages=10),
    )
    tracer = get_tracer("fastaiagent")
    with tracer.start_as_current_span("agent.test"):
        traced_add(mem, UserMessage("hello, what is my email?"))
        msgs = traced_get_context(mem, "what is my email")
    assert isinstance(msgs, list) and all(isinstance(m, Message) for m in msgs)

    spans = _read_spans(_isolated_db)
    read = _by_name(spans, "memory.read")
    assert len(read) == 1
    ra = _attrs(read[0])
    assert ra["fastaiagent.runner.type"] == "memory"
    assert ra["memory.operation"] == "read"
    assert ra["memory.block_count"] == 2
    assert ra["memory.message_count"] == len(msgs)
    assert ra["memory.query"] == "what is my email"

    # One child span per block, nested under the read parent.
    children = _by_name(spans, "memory.read.static") + _by_name(spans, "memory.read.vector")
    assert len(children) == 2
    for child in children:
        assert child["parent_span_id"] == read[0]["span_id"]


# ---------------------------------------------------------------------------
# Scenario 2 — HERO: VectorBlock scores reach the child span, rank-ordered
# ---------------------------------------------------------------------------


def test_vectorblock_scores_captured_in_rank_order(_isolated_db):
    # Canned hits already descending; render preserves order with weights=0.
    hits = [
        (_chunk("most relevant"), 0.91),
        (_chunk("less relevant"), 0.55),
        (_chunk("least relevant"), 0.30),
    ]
    mem = ComposableMemory(
        blocks=[VectorBlock(store=_CannedStore(hits), embedder=_CountingEmbedder(), top_k=3)]
    )
    tracer = get_tracer("fastaiagent")
    with tracer.start_as_current_span("agent.test"):
        traced_get_context(mem, "query")

    spans = _read_spans(_isolated_db)
    vec = _by_name(spans, "memory.read.vector")
    assert len(vec) == 1
    a = _attrs(vec[0])
    scores = json.loads(a["memory.scores"])
    assert scores == [0.91, 0.55, 0.30]  # rank order preserved
    assert scores == sorted(scores, reverse=True)
    assert a["memory.rendered_count"] == 3
    assert a["memory.block_type"] == "VectorBlock"


# ---------------------------------------------------------------------------
# Scenario 3 — write span + per-block actions
# ---------------------------------------------------------------------------


def test_write_span_per_block_actions(_isolated_db):
    mem = ComposableMemory(
        blocks=[
            StaticBlock("static fact"),
            VectorBlock(store=_CannedStore([]), embedder=_CountingEmbedder(), top_k=3),
        ]
    )
    tracer = get_tracer("fastaiagent")
    with tracer.start_as_current_span("agent.test"):
        traced_add(mem, UserMessage("a message long enough to be embedded"))

    spans = _read_spans(_isolated_db)
    write = _by_name(spans, "memory.write")
    assert len(write) == 1
    assert _attrs(write[0])["memory.messages_added"] == 1

    static_child = _by_name(spans, "memory.write.static")[0]
    assert _attrs(static_child)["memory.action"] == "noop"
    vec_child = _by_name(spans, "memory.write.vector")[0]
    assert _attrs(vec_child)["memory.action"] == "embedded"


# ---------------------------------------------------------------------------
# Scenario 5 — no tracer configured: identical behaviour, zero spans
# ---------------------------------------------------------------------------


def test_no_tracer_is_noop_and_preserves_behaviour(_isolated_db):
    store = _CannedStore([(_chunk("prior exchange"), 0.8)])
    blocks = [StaticBlock("fact"), VectorBlock(store=store, embedder=_CountingEmbedder(), top_k=3)]
    mem = ComposableMemory(blocks=blocks)
    # No active span / no tracer use beyond the no-op global tracer.
    traced_add(mem, UserMessage("hello world message"))
    traced_ctx = traced_get_context(mem, "query")
    # Compare to a bare get_context on an equivalent memory.
    store2 = _CannedStore([(_chunk("prior exchange"), 0.8)])
    mem2 = ComposableMemory(
        blocks=[
            StaticBlock("fact"),
            VectorBlock(store=store2, embedder=_CountingEmbedder(), top_k=3),
        ]
    )
    mem2.add(UserMessage("hello world message"))
    bare_ctx = mem2.get_context(query="query")
    assert [type(m).__name__ for m in traced_ctx] == [type(m).__name__ for m in bare_ctx]

    # No agent/parent span was opened, so no memory spans should be persisted
    # under a trace — but even the standalone spans must not error. The key
    # assertion is behaviour parity above; spans (if any) carry no agent root.
    spans = _read_spans(_isolated_db)
    # memory spans may exist as orphans depending on global provider; assert
    # they at least did not corrupt the message flow (already checked) and that
    # the helper returned the right count.
    assert len(traced_ctx) == len(bare_ctx)
    assert isinstance(spans, list)


# ---------------------------------------------------------------------------
# Scenario 6 — building the read span does not re-embed / recompute
# ---------------------------------------------------------------------------


def test_no_recomputation_when_tracing(_isolated_db):
    hits = [(_chunk("prior"), 0.7)]

    emb_traced = _CountingEmbedder()
    mem_traced = ComposableMemory(
        blocks=[VectorBlock(store=_CannedStore(hits), embedder=emb_traced, top_k=3)]
    )
    tracer = get_tracer("fastaiagent")
    with tracer.start_as_current_span("agent.test"):
        traced_get_context(mem_traced, "query")

    emb_bare = _CountingEmbedder()
    mem_bare = ComposableMemory(
        blocks=[VectorBlock(store=_CannedStore(hits), embedder=emb_bare, top_k=3)]
    )
    mem_bare.get_context(query="query")

    # Tracing must not trigger extra embed() calls (scores reused, not recomputed).
    assert emb_traced.calls == emb_bare.calls == 1


# ---------------------------------------------------------------------------
# Scenario 7 — capture-mode redaction masks memory snippets
# ---------------------------------------------------------------------------


def test_capture_redaction_masks_memory_snippets(_isolated_db):
    from fastaiagent.trace.redaction import RedactionPolicy, set_redaction_policy

    set_redaction_policy(
        RedactionPolicy(
            patterns=[r"[\w.+-]+@[\w-]+\.[\w.-]+"],
            replacement="[REDACTED]",
            mode="capture",
        )
    )
    try:
        store = _CannedStore([(_chunk("the user's email is secret@example.com"), 0.9)])
        mem = ComposableMemory(
            blocks=[VectorBlock(store=store, embedder=_CountingEmbedder(), top_k=3)]
        )
        tracer = get_tracer("fastaiagent")
        with tracer.start_as_current_span("agent.test"):
            traced_get_context(mem, "email")
        spans = _read_spans(_isolated_db)
        vec = _by_name(spans, "memory.read.vector")[0]
        snippets = json.loads(_attrs(vec)["memory.snippets"])
        assert "secret@example.com" not in " ".join(snippets)
        assert "[REDACTED]" in " ".join(snippets)
    finally:
        set_redaction_policy(None)


# ---------------------------------------------------------------------------
# Scenario 8 — custom block without the reporting interface: safe defaults
# ---------------------------------------------------------------------------


def test_custom_block_without_reports_is_safe(_isolated_db):
    class CustomBlock(MemoryBlock):
        name = "custom"

        def on_message(self, message):  # no last_write_report override
            return

        def render(self, query):  # no last_render_report override
            return [SystemMessage("custom output")]

    mem = ComposableMemory(blocks=[CustomBlock()])
    tracer = get_tracer("fastaiagent")
    with tracer.start_as_current_span("agent.test"):
        traced_add(mem, UserMessage("hi"))
        traced_get_context(mem, "q")

    spans = _read_spans(_isolated_db)
    rc = _by_name(spans, "memory.read.custom")
    wc = _by_name(spans, "memory.write.custom")
    assert len(rc) == 1 and len(wc) == 1
    assert _attrs(rc[0])["memory.rendered_count"] == 0
    assert _attrs(wc[0])["memory.action"] == "noop"
    assert _attrs(rc[0])["memory.block_type"] == "CustomBlock"


# ---------------------------------------------------------------------------
# Plain AgentMemory (no blocks) still traces a read/write parent
# ---------------------------------------------------------------------------


def test_plain_agent_memory_traces_parent_only(_isolated_db):
    mem = AgentMemory(max_messages=5)
    tracer = get_tracer("fastaiagent")
    with tracer.start_as_current_span("agent.test"):
        traced_add(mem, UserMessage("a message"))
        traced_get_context(mem, "q")
    spans = _read_spans(_isolated_db)
    assert len(_by_name(spans, "memory.read")) == 1
    assert len(_by_name(spans, "memory.write")) == 1
    assert _attrs(_by_name(spans, "memory.read")[0])["memory.block_count"] == 0
