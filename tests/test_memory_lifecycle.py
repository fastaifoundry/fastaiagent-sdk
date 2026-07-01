"""Full fact lifecycle via the Memory facade: create → update → supersede → remove.

Deterministic, real SQLite store (no mocking, no LLM). Supersede + delete are
also verified across Postgres/Redis in tests/integration/test_faststore_conformance.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastaiagent._internal.config import reset_config
from fastaiagent.agent.memory_simple import Memory
from fastaiagent.learn import MemoryStore


@pytest.fixture
def mem(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    reset_config()
    yield Memory(location=MemoryStore(db_path=str(tmp_path / "facts.db")))
    reset_config()


def test_create_then_retrieve(mem: Memory):
    mem.persist("Alice prefers email", tier="user", id="alice")
    assert [f.fact for f in mem.retrieve(tier="user", id="alice")] == ["Alice prefers email"]


def test_update_supersedes_old_by_text(mem: Memory):
    mem.persist("Alice prefers email", tier="user", id="alice")
    new_id = mem.update("Alice prefers Slack", old="Alice prefers email", tier="user", id="alice")
    active = [f.fact for f in mem.retrieve(tier="user", id="alice")]
    assert active == ["Alice prefers Slack"]  # old hidden, new active
    assert isinstance(new_id, int)


def test_update_by_id_preserves_history(mem: Memory):
    old_id = mem.persist("v1", tier="user", id="u")
    new_id = mem.update("v2", old=old_id, tier="user", id="u")
    # old row still exists but is marked superseded (audit history preserved)
    store = mem._store
    old_row = store.get(old_id)
    assert old_row is not None and old_row.superseded_by == new_id
    assert [f.fact for f in mem.retrieve(tier="user", id="u")] == ["v2"]


def test_update_missing_old_raises(mem: Memory):
    with pytest.raises(ValueError, match="no active fact matching"):
        mem.update("new", old="does-not-exist", tier="user", id="alice")


def test_update_requires_user_id(mem: Memory):
    with pytest.raises(ValueError, match="requires id"):
        mem.update("x", old="y", tier="user")


def test_remove_deletes_fact(mem: Memory):
    mem.persist("temp fact", tier="user", id="alice")
    assert mem.forget(tier="user", id="alice", fact="temp fact") == 1
    assert mem.retrieve(tier="user", id="alice") == []


def test_remove_all_for_subject_incl_superseded(mem: Memory):
    mem.persist("v1", tier="user", id="alice")
    mem.update("v2", old="v1", tier="user", id="alice")
    # forget the whole subject → removes active AND superseded history
    removed = mem.forget(tier="user", id="alice")
    assert removed >= 2
    assert mem.retrieve(tier="user", id="alice") == []


def test_remove_refuses_without_id(mem: Memory):
    mem.persist("x", tier="user", id="alice")
    with pytest.raises(ValueError):
        mem.forget(tier="user")  # empty id at user scope → refuse mass-delete


def test_update_emits_span(tmp_path: Path, monkeypatch):

    from fastaiagent._internal.storage import SQLiteHelper
    from fastaiagent.trace.otel import get_tracer
    from fastaiagent.trace.otel import reset as reset_tracer

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    reset_config()
    reset_tracer()
    mem = Memory(location=MemoryStore(db_path=str(db)))
    tracer = get_tracer("fastaiagent")
    with tracer.start_as_current_span("agent.test"):
        mem.persist("v1", tier="user", id="u")
        mem.update("v2", old="v1", tier="user", id="u")
    reset_tracer()
    with SQLiteHelper(db) as d:
        names = {r["name"] for r in d.fetchall("SELECT name FROM spans")}
    assert "memory.update" in names
    reset_config()
