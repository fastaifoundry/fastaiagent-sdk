"""FactStore conformance — the SAME contract across SQLite / Postgres / Redis.

No mocking: real backends. SQLite always runs; Postgres and Redis are gated on
``PG_TEST_DSN`` / ``REDIS_TEST_URL``, which ``tests/integration/conftest.py``
fills in for you when the dev containers are up::

    scripts/dev_backends.sh up
    pytest tests/integration/test_faststore_conformance.py

Set either variable yourself to target a different server; an explicit value
always wins. Until the ``durability-backends`` CI job existed, the Postgres and
Redis parameters of this file ran nowhere — no job set ``REDIS_TEST_URL`` and
this file was absent from the one step that set ``PG_TEST_DSN``.

Tests use uuid-suffixed scope_ids so runs against a shared server don't collide.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from fastaiagent.learn import Fact, FactStore

PG_DSN = os.environ.get("PG_TEST_DSN")
REDIS_URL = os.environ.get("REDIS_TEST_URL")


@pytest.fixture(params=["sqlite", "postgres", "redis"])
def store(request: pytest.FixtureRequest, tmp_path) -> Iterator[FactStore]:
    backend = request.param
    if backend == "sqlite":
        from fastaiagent.learn import MemoryStore

        yield MemoryStore(db_path=str(tmp_path / "facts.db"))
    elif backend == "postgres":
        if not PG_DSN:
            pytest.skip("PG_TEST_DSN not set")
        from fastaiagent.learn import PostgresFactStore

        yield PostgresFactStore(PG_DSN)
    elif backend == "redis":
        if not REDIS_URL:
            pytest.skip("REDIS_TEST_URL not set")
        from fastaiagent.learn import RedisFactStore

        yield RedisFactStore(REDIS_URL, namespace=f"t{uuid.uuid4().hex[:8]}")


@pytest.fixture
def uid() -> str:
    return f"u-{uuid.uuid4().hex[:8]}"


def test_conforms_to_protocol(store: FactStore):
    assert isinstance(store, FactStore)


def test_add_is_idempotent(store: FactStore, uid: str):
    a = store.add(Fact(scope="user", scope_id=uid, fact="likes email"))
    b = store.add(Fact(scope="user", scope_id=uid, fact="likes email"))
    assert a == b
    assert [f.fact for f in store.list_active(scope="user", scope_id=uid)] == ["likes email"]


def test_get_roundtrip(store: FactStore, uid: str):
    fid = store.add(Fact(scope="user", scope_id=uid, fact="hi", confidence=0.6))
    got = store.get(fid)
    assert got is not None and got.fact == "hi" and got.confidence == 0.6


def test_safe_default_empty_scope_id(store: FactStore, uid: str):
    store.add(Fact(scope="user", scope_id=uid, fact="secret"))
    assert store.list_active(scope="user", scope_id="") == []  # never leak
    assert [f.fact for f in store.list_active(scope="user", scope_id=uid)] == ["secret"]


def test_star_matches_all_within_scope(store: FactStore):
    a, b = f"a-{uuid.uuid4().hex[:6]}", f"b-{uuid.uuid4().hex[:6]}"
    store.add(Fact(scope="user", scope_id=a, fact="fact-a"))
    store.add(Fact(scope="user", scope_id=b, fact="fact-b"))
    facts = {f.fact for f in store.list_active(scope="user", scope_id="*")}
    assert {"fact-a", "fact-b"} <= facts


def test_agent_scope_permissive(store: FactStore):
    sid = f"ag-{uuid.uuid4().hex[:6]}"
    store.add(Fact(scope="agent", scope_id=sid, fact="global-x"))
    # agent scope: exact id works; empty id is permissive (returns rows)
    assert [f.fact for f in store.list_active(scope="agent", scope_id=sid)] == ["global-x"]
    assert "global-x" in {f.fact for f in store.list_active(scope="agent", scope_id="")}


def test_supersede_hides_old_keeps_new(store: FactStore, uid: str):
    old = store.add(Fact(scope="user", scope_id=uid, fact="v1"))
    new = store.add(Fact(scope="user", scope_id=uid, fact="v2"))
    store.supersede(old, new)
    active = [f.fact for f in store.list_active(scope="user", scope_id=uid)]
    assert "v2" in active and "v1" not in active


def test_delete_guard_and_full_delete(store: FactStore, uid: str):
    old = store.add(Fact(scope="user", scope_id=uid, fact="v1"))
    new = store.add(Fact(scope="user", scope_id=uid, fact="v2"))
    store.supersede(old, new)
    with pytest.raises(ValueError):
        store.delete(scope="user", scope_id="")  # refuse mass-delete
    # forget the subject → removes ALL rows incl. superseded history
    store.delete(scope="user", scope_id=uid)
    assert store.list_active(scope="user", scope_id=uid) == []
    assert store.list_active(scope="user", scope_id="*") == [] or uid not in {
        f.scope_id for f in store.list_active(scope="user", scope_id="*")
    }


def test_project_partition_isolates(store: FactStore, uid: str):
    store.add(Fact(scope="user", scope_id=uid, fact="in-A", project_id="A"))
    store.add(Fact(scope="user", scope_id=uid, fact="in-B", project_id="B"))
    assert [f.fact for f in store.list_active(scope="user", scope_id=uid, project_id="A")] == [
        "in-A"
    ]
    assert [f.fact for f in store.list_active(scope="user", scope_id=uid, project_id="B")] == [
        "in-B"
    ]
