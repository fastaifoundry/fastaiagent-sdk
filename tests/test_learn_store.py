"""Unit tests for fastaiagent.learn.store — no mocking, isolated DB."""

from __future__ import annotations

import pytest

from fastaiagent.learn import Fact, MemoryStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Construct a MemoryStore against a fresh local.db in tmp_path."""
    from fastaiagent._internal.config import reset_config

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    reset_config()
    yield MemoryStore()
    reset_config()


# ─── Migration / table creation ─────────────────────────────────────────────


def test_store_construction_runs_migration(store) -> None:
    """Construction should create the learned_memory table via init_local_db."""
    facts = store.list_active(scope="agent", scope_id="any")
    assert facts == []  # no rows but no error → table exists


def test_schema_v8_user_version(tmp_path, monkeypatch) -> None:
    """Construction bumps PRAGMA user_version to at least 8."""
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    MemoryStore()  # runs migration

    from fastaiagent._internal.storage import SQLiteHelper

    db = SQLiteHelper(str(tmp_path / "local.db"))
    row = db.fetchone("PRAGMA user_version")
    db.close()
    assert row is not None
    assert int(next(iter(row.values()))) >= 8


# ─── Insert + dedup ─────────────────────────────────────────────────────────


def test_add_returns_id(store) -> None:
    fid = store.add(Fact(scope="agent", scope_id="x", fact="hello"))
    assert fid > 0


def test_add_idempotent_on_duplicate(store) -> None:
    f = Fact(scope="agent", scope_id="x", fact="dup test")
    a = store.add(f)
    b = store.add(f)
    assert a == b
    assert len(store.list_active(scope="agent", scope_id="x")) == 1


def test_add_rejects_empty_fact(store) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        store.add(Fact(scope="agent", scope_id="x", fact="   "))


def test_add_rejects_bad_scope(store) -> None:
    with pytest.raises(ValueError, match="scope"):
        store.add(Fact(scope="other", scope_id="x", fact="x"))  # type: ignore[arg-type]


def test_add_many(store) -> None:
    ids = store.add_many(
        [
            Fact(scope="agent", scope_id="x", fact="one"),
            Fact(scope="agent", scope_id="x", fact="two"),
            Fact(scope="agent", scope_id="x", fact="three"),
        ]
    )
    assert len(ids) == 3
    assert len(set(ids)) == 3
    assert len(store.list_active(scope="agent", scope_id="x")) == 3


# ─── List filters ───────────────────────────────────────────────────────────


def test_list_active_filters_by_scope_id(store) -> None:
    store.add(Fact(scope="agent", scope_id="a", fact="for-a"))
    store.add(Fact(scope="agent", scope_id="b", fact="for-b"))
    a_facts = store.list_active(scope="agent", scope_id="a")
    b_facts = store.list_active(scope="agent", scope_id="b")
    assert {f.fact for f in a_facts} == {"for-a"}
    assert {f.fact for f in b_facts} == {"for-b"}


def test_list_active_excludes_superseded(store) -> None:
    old = store.add(Fact(scope="agent", scope_id="x", fact="old"))
    new = store.add(Fact(scope="agent", scope_id="x", fact="new"))
    store.supersede(old, new)
    facts = store.list_active(scope="agent", scope_id="x")
    assert {f.fact for f in facts} == {"new"}


def test_list_all_includes_superseded(store) -> None:
    old = store.add(Fact(scope="agent", scope_id="x", fact="old"))
    new = store.add(Fact(scope="agent", scope_id="x", fact="new"))
    store.supersede(old, new)
    all_facts = store.list_all()
    assert len(all_facts) == 2


def test_list_active_orders_newest_first(store) -> None:
    import time

    store.add(Fact(scope="agent", scope_id="x", fact="first", created_at=time.time()))
    store.add(
        Fact(scope="agent", scope_id="x", fact="second", created_at=time.time() + 1)
    )
    facts = store.list_active(scope="agent", scope_id="x")
    assert facts[0].fact == "second"
    assert facts[1].fact == "first"


# ─── Conflict resolution ────────────────────────────────────────────────────


def test_supersede_marks_chain(store) -> None:
    old = store.add(Fact(scope="agent", scope_id="x", fact="A"))
    new = store.add(Fact(scope="agent", scope_id="x", fact="B"))
    store.supersede(old, new)
    old_row = store.get(old)
    assert old_row is not None
    assert old_row.superseded_by == new


def test_supersede_missing_row_raises(store) -> None:
    with pytest.raises(ValueError, match="missing"):
        store.supersede(99999, 99998)


def test_supersede_idempotent(store) -> None:
    old = store.add(Fact(scope="agent", scope_id="x", fact="A"))
    new = store.add(Fact(scope="agent", scope_id="x", fact="B"))
    store.supersede(old, new)
    store.supersede(old, new)  # second call should be a no-op, not crash
    assert store.get(old).superseded_by == new


# ─── Project scoping ────────────────────────────────────────────────────────


def test_project_id_isolates_facts(store) -> None:
    store.add(Fact(scope="agent", scope_id="x", fact="proj-a", project_id="A"))
    store.add(Fact(scope="agent", scope_id="x", fact="proj-b", project_id="B"))
    a = store.list_active(scope="agent", scope_id="x", project_id="A")
    b = store.list_active(scope="agent", scope_id="x", project_id="B")
    assert {f.fact for f in a} == {"proj-a"}
    assert {f.fact for f in b} == {"proj-b"}
