"""``PostgresFactStore`` reuses its connections (1.82.0).

It opened a new connection for every call: one per turn for the fact read,
one per learned fact, one per hit in a semantic search. It now holds a small
pool, per process, so a store built before a fork is still safe to use in
the child.

No mocking: a real Postgres, gated on ``PG_TEST_DSN`` (see
``tests/integration/conftest.py``).
"""

from __future__ import annotations

import os
import signal
import sys
import time
import uuid

import pytest

PG_DSN = os.environ.get("PG_TEST_DSN")

pytestmark = pytest.mark.skipif(not PG_DSN, reason="PG_TEST_DSN not set")


def _tagged(dsn: str, app: str) -> str:
    return f"{dsn}{'&' if '?' in dsn else '?'}application_name={app}"


def _monitor():
    import psycopg

    return psycopg.connect(PG_DSN, autocommit=True)


def _sessions_opened(conn) -> int:
    conn.execute("SELECT pg_stat_clear_snapshot()")
    row = conn.execute(
        "SELECT sessions FROM pg_stat_database WHERE datname = current_database()"
    ).fetchone()
    return int(row[0])


def _open_connections(conn, app: str) -> int:
    row = conn.execute(
        "SELECT count(*) FROM pg_stat_activity WHERE application_name = %s", (app,)
    ).fetchone()
    return int(row[0])


def test_a_store_reuses_its_connections():
    from fastaiagent.learn import Fact, PostgresFactStore

    subject = f"pool-{uuid.uuid4().hex[:8]}"
    store = PostgresFactStore(PG_DSN, max_pool_size=2)
    with _monitor() as mon:
        before = _sessions_opened(mon)
        ids = [store.add(Fact(scope="user", scope_id=subject, fact=f"f{i}")) for i in range(10)]
        for _ in range(20):
            store.list_active("user", subject)
        for fid in ids:
            store.get(fid)
        store.supersede(ids[0], ids[1])
        store.delete("user", subject)
        opened = _sessions_opened(mon) - before
    store.close()
    assert opened <= 2  # was one per call: 42 here


def test_close_releases_the_connections_and_the_store_reopens():
    from fastaiagent.learn import PostgresFactStore

    app = f"fa-pool-{uuid.uuid4().hex[:8]}"
    store = PostgresFactStore(_tagged(PG_DSN, app))
    with _monitor() as mon:
        store.list_active("agent", "pool-probe")
        assert _open_connections(mon, app) >= 1
        store.close()
        assert _open_connections(mon, app) == 0
        store.close()  # idempotent
        assert store.list_active("agent", "pool-probe") == []  # reopens on use
        store.close()


def test_a_bad_dsn_still_fails_at_construction_with_the_real_error():
    """A pool connects in the background, so a bad DSN would only surface as a
    pool timeout on first use; construction must keep failing fast."""
    import psycopg

    from fastaiagent.learn import PostgresFactStore

    bad = PG_DSN.replace("postgres:test@", "postgres:wrong-password@")
    started = time.monotonic()
    with pytest.raises(psycopg.OperationalError):
        PostgresFactStore(bad)
    assert time.monotonic() - started < 10


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork and SIGALRM")
def test_a_forked_child_never_uses_the_parents_connections():
    """A store built before a fork (``gunicorn --preload``) must not hand the
    parent's sockets to the child: both sides would talk over one connection."""
    import warnings

    from fastaiagent.learn import Fact, PostgresFactStore

    subject = f"fork-{uuid.uuid4().hex[:8]}"
    store = PostgresFactStore(PG_DSN, min_pool_size=1, max_pool_size=1)
    store.add(Fact(scope="user", scope_id=subject, fact="shared fact"))  # parent pool is live

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # fork with threads
        pid = os.fork()
    if pid == 0:  # child
        signal.alarm(20)  # a shared connection hangs; the default action kills us
        code = 0
        try:
            for _ in range(200):
                if [f.fact for f in store.list_active("user", subject)] != ["shared fact"]:
                    code = 1
                    break
        except BaseException:
            code = 2
        finally:
            sys.stdout.flush()
            os._exit(code)

    def _stuck(signum, frame):
        raise TimeoutError("parent stuck on a connection shared with the child")

    previous = signal.signal(signal.SIGALRM, _stuck)
    signal.alarm(30)
    parent_ok = True
    try:
        for _ in range(200):
            if [f.fact for f in store.list_active("user", subject)] != ["shared fact"]:
                parent_ok = False
                break
    except TimeoutError:
        parent_ok = False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        _, status = os.waitpid(pid, 0)
    child_code = os.waitstatus_to_exitcode(status)
    store.delete("user", subject)
    store.close()
    assert child_code == 0 and parent_ok


def test_a_dropped_store_closes_its_pool_cleanly():
    """A store dropped without ``close()`` left its pool to ``psycopg_pool``'s
    own ``__del__``, which could run on one of the pool's worker threads and
    print "RuntimeError: cannot join current thread". The store now closes its
    pool the moment it is collected."""
    import gc

    from fastaiagent.learn import PostgresFactStore

    app = f"fa-drop-{uuid.uuid4().hex[:8]}"
    caught: list[str] = []
    previous = sys.unraisablehook
    sys.unraisablehook = lambda u: caught.append(f"{type(u.exc_value).__name__}: {u.exc_value}")
    try:
        for _ in range(20):
            store = PostgresFactStore(_tagged(PG_DSN, app), min_pool_size=3, max_pool_size=4)
            store.list_active("agent", "pool-probe")  # workers still opening connections
            del store  # no close()
            gc.collect()
        time.sleep(1.0)
        gc.collect()
    finally:
        sys.unraisablehook = previous
    assert caught == []
    with _monitor() as mon:
        assert _open_connections(mon, app) == 0
