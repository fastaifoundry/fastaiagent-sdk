"""Redis fact reads cost what they return, not what is stored (1.80.0).

``PersistentFactBlock`` reads the newest ``max_facts`` facts on every turn.
``RedisFactStore.list_active`` used to fetch every matching fact — one
``HGETALL`` round trip each — and only then sort and slice. A user whose
``learn=`` facts piled up made every one of their turns slower, and put that
load on a Redis shared by everyone; an agent-wide read (``scope_id=""``) paid
for every agent's facts. Reads now go through a newest-first sorted index.

No mocking: a real Redis, gated on ``REDIS_TEST_URL`` (``tests/integration/
conftest.py`` fills it in when ``scripts/dev_backends.sh up`` is running). The
work bound is measured with Redis's own per-command call counters.
"""

from __future__ import annotations

import os
import uuid

import pytest

from fastaiagent.learn import Fact

REDIS_URL = os.environ.get("REDIS_TEST_URL")

pytestmark = pytest.mark.skipif(not REDIS_URL, reason="REDIS_TEST_URL not set")


@pytest.fixture
def store():
    from fastaiagent.learn import RedisFactStore

    return RedisFactStore(REDIS_URL, namespace=f"t{uuid.uuid4().hex[:8]}")


def _fact_fetches(r) -> int:
    """Server-wide ``HGETALL`` calls so far — one per fact hash read."""
    return int(r.info("commandstats").get("cmdstat_hgetall", {}).get("calls", 0))


def test_a_subjects_read_fetches_only_limit_facts(store):
    for i in range(300):
        store.add(Fact(scope="user", scope_id="heavy", fact=f"fact {i}", created_at=1000.0 + i))

    before = _fact_fetches(store._r)
    got = store.list_active(scope="user", scope_id="heavy", limit=5)

    assert [f.fact for f in got] == [f"fact {i}" for i in range(299, 294, -1)]
    assert _fact_fetches(store._r) - before == 5


def test_an_agent_wide_read_fetches_only_limit_facts(store):
    for a in range(40):
        for i in range(5):
            store.add(
                Fact(
                    scope="agent", scope_id=f"agent-{a}", fact=f"a{a} f{i}", created_at=a * 10.0 + i
                )
            )

    before = _fact_fetches(store._r)
    got = store.list_active(scope="agent", scope_id="", limit=3)

    assert [f.fact for f in got] == ["a39 f4", "a39 f3", "a39 f2"]
    assert _fact_fetches(store._r) - before == 3


def test_facts_written_by_an_older_sdk_are_indexed_on_open(store):
    """A namespace written before the sorted index existed has only the
    active-id sets. Opening it with this version indexes those facts once."""
    from fastaiagent.learn import RedisFactStore

    for i in range(3):
        store.add(Fact(scope="user", scope_id="u", fact=f"old {i}", created_at=500.0 + i))
    # Reduce the namespace to what an older SDK wrote: no sorted indexes and no
    # schema marker.
    for key in store._r.scan_iter(match=f"{store._ns}:z*"):
        store._r.delete(key)
    store._r.delete(f"{store._ns}:schema")

    reopened = RedisFactStore(REDIS_URL, namespace=store._ns)
    got = reopened.list_active(scope="user", scope_id="u", limit=2)
    assert [f.fact for f in got] == ["old 2", "old 1"]
