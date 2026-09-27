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


# The per-turn read: PersistentFactBlock asks for the newest ``max_facts``.


def test_limit_returns_the_newest_first(store: FactStore, uid: str):
    for i in range(12):
        store.add(Fact(scope="user", scope_id=uid, fact=f"fact-{i:02d}", created_at=1000.0 + i))
    got = [f.fact for f in store.list_active(scope="user", scope_id=uid, limit=3)]
    assert got == ["fact-11", "fact-10", "fact-09"]


def test_limit_skips_superseded_facts(store: FactStore, uid: str):
    ids = [
        store.add(Fact(scope="user", scope_id=uid, fact=f"v{i}", created_at=2000.0 + i))
        for i in range(4)
    ]
    store.supersede(ids[3], ids[2])  # the newest row is superseded
    got = [f.fact for f in store.list_active(scope="user", scope_id=uid, limit=2)]
    assert got == ["v2", "v1"]


def test_wildcard_limit_is_newest_first_across_subjects(store: FactStore):
    project = f"p-{uuid.uuid4().hex[:8]}"  # isolates this test on a shared server
    for i, sid in enumerate(["ag-a", "ag-b", "ag-c"] * 2):
        store.add(
            Fact(
                scope="agent", scope_id=sid, fact=f"g{i}", created_at=3000.0 + i, project_id=project
            )
        )
    for sid in ("*", ""):  # "" at agent scope is the permissive (all-agents) read
        got = store.list_active(scope="agent", scope_id=sid, project_id=project, limit=2)
        assert [f.fact for f in got] == ["g5", "g4"]


def test_concurrent_adds_of_one_fact_keep_one_row(store: FactStore, uid: str):
    """``add`` looked for the fact, then inserted it. Writers that raced between
    the two either failed on the unique constraint (SQLite, Postgres) or both
    succeeded and stored the fact twice (Redis). Every writer must get the one
    row's id."""
    import threading

    writers, rounds = 8, 5
    for r in range(rounds):
        fact = Fact(scope="user", scope_id=uid, fact=f"prefers email #{r}")
        gate = threading.Barrier(writers)
        ids: list[int] = []
        errors: list[BaseException] = []

        def write(fact: Fact = fact, gate: threading.Barrier = gate) -> None:
            gate.wait()
            try:
                ids.append(store.add(fact))
            except BaseException as err:  # noqa: BLE001 — recorded, asserted below
                errors.append(err)

        threads = [threading.Thread(target=write) for _ in range(writers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], f"round {r}: {errors[:2]}"
        assert len(set(ids)) == 1, f"round {r}: {sorted(set(ids))}"
    facts = [f.fact for f in store.list_active(scope="user", scope_id=uid)]
    assert sorted(facts) == sorted(f"prefers email #{r}" for r in range(rounds))


def test_source_round_trips(store: FactStore, uid: str):
    """A learned fact says so on its own (1.82.0): it used to be told apart only
    by a trace id, which it lacks when tracing is off."""
    store.add(Fact(scope="user", scope_id=uid, fact="learned one", source="learned"))
    store.add(Fact(scope="user", scope_id=uid, fact="yours"))
    by_fact = {f.fact: f.source for f in store.list_active(scope="user", scope_id=uid)}
    assert by_fact == {"learned one": "learned", "yours": ""}
