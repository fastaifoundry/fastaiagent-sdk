"""Memory facade + safe-by-default scoping + dynamic scope_id.

No mocking: real SQLite ``MemoryStore``, real blocks, real tracing. No LLM
(the facade's `persist` is verbatim; extraction is covered by the e2e gate).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from fastaiagent._internal.config import reset_config
from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.agent.context import (
    RunContext,
    reset_active_run_context,
    set_active_run_context,
)
from fastaiagent.agent.memory_blocks import PersistentFactBlock
from fastaiagent.agent.memory_simple import Memory
from fastaiagent.learn import Fact, MemoryStore
from fastaiagent.llm.message import UserMessage
from fastaiagent.trace.otel import get_tracer
from fastaiagent.trace.otel import reset as reset_tracer


@pytest.fixture
def db(tmp_path: Path, monkeypatch):
    p = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(p))
    reset_config()
    reset_tracer()
    yield p
    reset_tracer()
    reset_config()


@dataclass
class St:
    user_id: str


# ---------------------------------------------------------------------------
# Safe-by-default scoping (the leak fix)
# ---------------------------------------------------------------------------


def test_safe_default_user_scope(db):
    s = MemoryStore(db_path=str(db))
    s.add(Fact(scope="user", scope_id="alice", fact="a-fact"))
    s.add(Fact(scope="user", scope_id="bob", fact="b-fact"))
    # empty id at user scope → nothing (was: everyone)
    assert s.list_active(scope="user", scope_id="") == []
    # explicit "*" → all
    assert {f.fact for f in s.list_active(scope="user", scope_id="*")} == {"a-fact", "b-fact"}
    # specific id → that subject only
    assert [f.fact for f in s.list_active(scope="user", scope_id="alice")] == ["a-fact"]


def test_agent_scope_stays_permissive(db):
    s = MemoryStore(db_path=str(db))
    s.add(Fact(scope="agent", scope_id="x", fact="global-1"))
    s.add(Fact(scope="agent", scope_id="y", fact="global-2"))
    # agent is the global tier: empty id = all (unchanged)
    assert {f.fact for f in s.list_active(scope="agent", scope_id="")} == {"global-1", "global-2"}


def test_delete_refuses_mass_delete_without_id(db):
    s = MemoryStore(db_path=str(db))
    s.add(Fact(scope="user", scope_id="alice", fact="keep"))
    with pytest.raises(ValueError, match="explicit scope_id"):
        s.delete(scope="user", scope_id="")
    assert s.list_active(scope="user", scope_id="alice")  # untouched


# ---------------------------------------------------------------------------
# Dynamic scope_id + cross-session isolation (the multi-user guard)
# ---------------------------------------------------------------------------


def test_dynamic_scope_id_isolates_users(db):
    s = MemoryStore(db_path=str(db))
    s.add(Fact(scope="user", scope_id="alice", fact="Alice likes email"))
    s.add(Fact(scope="user", scope_id="bob", fact="Bob likes SMS"))
    block = PersistentFactBlock(scope="user", scope_id=lambda ctx: ctx.state.user_id, store=s)

    def render_for(uid: str) -> str:
        tok = set_active_run_context(RunContext(state=St(user_id=uid)))
        try:
            out = block.render("?")
            return out[0].content if out else ""
        finally:
            reset_active_run_context(tok)

    assert "Alice likes email" in render_for("alice")
    assert "Bob likes SMS" in render_for("bob")
    # switching back must NOT serve bob's cached facts (cache invalidation)
    assert "Alice likes email" in render_for("alice")
    assert "Bob" not in render_for("alice")


def test_dynamic_scope_id_no_context_is_safe(db):
    s = MemoryStore(db_path=str(db))
    s.add(Fact(scope="user", scope_id="alice", fact="secret"))
    block = PersistentFactBlock(scope="user", scope_id=lambda ctx: ctx.state.user_id, store=s)
    # No active RunContext → resolver yields "" → no personal facts.
    assert block.render("?") == []


# ---------------------------------------------------------------------------
# Memory facade — direct verbs
# ---------------------------------------------------------------------------


def test_persist_retrieve_forget_roundtrip(db):
    mem = Memory(location=MemoryStore(db_path=str(db)), agent_id="support")
    mem.persist("Return policy is 30 days", tier="global")
    fid = mem.persist("Alice prefers email", tier="user", id="alice")
    assert isinstance(fid, int)
    assert [f.fact for f in mem.retrieve(tier="global")] == ["Return policy is 30 days"]
    assert [f.fact for f in mem.retrieve(tier="user", id="alice")] == ["Alice prefers email"]
    # safe: user tier without id → []
    assert mem.retrieve(tier="user") == []
    # forget
    assert mem.forget(tier="user", id="alice") == 1
    assert mem.retrieve(tier="user", id="alice") == []


def test_persist_returns_fact_objects_with_metadata(db):
    mem = Memory(location=MemoryStore(db_path=str(db)))
    mem.persist("fact one", tier="user", id="u", confidence=0.6)
    rows = mem.retrieve(tier="user", id="u")
    assert isinstance(rows[0], Fact)
    assert rows[0].confidence == 0.6
    assert rows[0].source_trace_id is None  # direct persist = "manual"


def test_tier_validation(db):
    mem = Memory(location=MemoryStore(db_path=str(db)))
    with pytest.raises(ValueError, match="requires id"):
        mem.persist("x", tier="user")
    with pytest.raises(NotImplementedError):
        mem.retrieve(tier="session")
    with pytest.raises(NotImplementedError):
        mem.retrieve("semantic?", tier="user", id="u")
    with pytest.raises(ValueError, match="tier must be"):
        mem.retrieve(tier="bogus")


# ---------------------------------------------------------------------------
# Memory facade — agent-attachable contract
# ---------------------------------------------------------------------------


def test_memory_is_drop_in_window(db):
    mem = Memory(location=MemoryStore(db_path=str(db)), window=10)
    mem.add(UserMessage("hello"))
    ctx = mem.get_context("?")
    assert [m.content for m in ctx] == ["hello"]
    assert len(mem) == 1
    assert bool(mem) is True


def test_bare_memory_has_no_fact_blocks(db):
    mem = Memory(location=MemoryStore(db_path=str(db)))  # no user_id/agent_id
    assert mem.blocks == []  # window only, no surprise DB reads


def test_dynamic_memory_routes_per_user_no_window_leak(db):
    """One Memory, dynamic user_id: durable facts AND the live window are
    isolated per user (the multi-session guarantee)."""
    store = MemoryStore(db_path=str(db))
    store.add(Fact(scope="user", scope_id="alice", fact="Alice fact"))
    store.add(Fact(scope="user", scope_id="bob", fact="Bob fact"))
    mem = Memory(location=store, user_id=lambda ctx: ctx.state.user_id)

    def run_turn(uid: str, text: str) -> list[str]:
        tok = set_active_run_context(RunContext(state=St(user_id=uid)))
        try:
            mem.add(UserMessage(text))  # goes into THIS user's window
            return [m.content or "" for m in mem.get_context("?")]
        finally:
            reset_active_run_context(tok)

    alice_ctx = run_turn("alice", "alice-private-message")
    bob_ctx = run_turn("bob", "bob-private-message")

    joined_alice = " ".join(alice_ctx)
    joined_bob = " ".join(bob_ctx)
    # durable facts isolated
    assert "Alice fact" in joined_alice and "Bob fact" not in joined_alice
    assert "Bob fact" in joined_bob and "Alice fact" not in joined_bob
    # live window isolated — bob never sees alice's message and vice versa
    assert "alice-private-message" not in joined_bob
    assert "bob-private-message" not in joined_alice
    # switching back to alice: her window persists, no bob bleed
    alice_again = run_turn("alice", "second-alice-message")
    j = " ".join(alice_again)
    assert "alice-private-message" in j and "bob-private-message" not in j


# ---------------------------------------------------------------------------
# Direct-op spans
# ---------------------------------------------------------------------------


def _read_spans(db_path: Path):
    reset_tracer()
    with SQLiteHelper(db_path) as d:
        return d.fetchall("SELECT name, attributes FROM spans ORDER BY start_time")


def test_persist_and_retrieve_emit_spans(db):
    mem = Memory(location=MemoryStore(db_path=str(db)), agent_id="support")
    tracer = get_tracer("fastaiagent")
    with tracer.start_as_current_span("agent.test"):
        mem.persist("g", tier="global")
        mem.retrieve(tier="global")

    spans = {r["name"]: json.loads(r["attributes"]) for r in _read_spans(db)}
    assert "memory.persist" in spans
    assert "memory.retrieve" in spans
    assert spans["memory.persist"]["memory.tier"] == "global"
    assert spans["memory.persist"]["memory.scope"] == "agent"
    assert spans["memory.persist"]["memory.count"] == 1
    assert spans["memory.retrieve"]["memory.count"] == 1
