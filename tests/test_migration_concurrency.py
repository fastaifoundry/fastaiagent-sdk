"""Two processes opening one ``local.db`` must not collide on migrations.

This is not a hypothetical. It took down a CI gate as

    sqlite3.OperationalError: trigger spans_fts_ai already exists

reported by the test as "worker exited with 1 before checkpointing step_2",
which named the symptom and hid the cause. It is reachable in the ordinary
local setup, because the UI server and an agent run share one ``local.db``.

The race: ``_run_migrations`` read ``PRAGMA user_version``, then applied steps,
and every statement autocommitted. A second process could read the version,
decide the migration was outstanding, and collide with the schema the first had
already changed. ``v10`` is the one that bites, because it *drops* three
triggers and recreates them — so the loser finds them either missing or already
present depending on where it lands in the winner's sequence.
"""

from __future__ import annotations

import multiprocessing as mp
import sqlite3

import pytest

from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.ui.db import _run_migrations, init_local_db

# ``fork`` inherits the parent's SQLite connections and module state, which
# hides exactly the cross-process behaviour under test. ``spawn`` starts clean.
_CTX = mp.get_context("spawn")


def _init_in_child(db_path: str, barrier: object, errors: object) -> None:  # pragma: no cover
    """Wait at the barrier so every process migrates at the same instant."""
    try:
        barrier.wait(timeout=30)  # type: ignore[attr-defined]
        init_local_db(db_path)
    except BaseException as exc:  # noqa: BLE001 — the failure IS the result
        errors.put(f"{type(exc).__name__}: {exc}")  # type: ignore[attr-defined]


@pytest.mark.parametrize("workers", [4])
def test_concurrent_init_local_db_does_not_collide(tmp_path, workers: int) -> None:
    """The real reproduction: N processes, one fresh database, one instant.

    Red before the fix — at least one child dies with
    ``trigger spans_fts_ai already exists``. The barrier is what makes it
    reliable; without it the processes queue up and the bug hides.
    """
    db_path = str(tmp_path / "local.db")
    barrier = _CTX.Barrier(workers)
    errors: mp.Queue = _CTX.Queue()

    procs = [
        _CTX.Process(target=_init_in_child, args=(db_path, barrier, errors))
        for _ in range(workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)

    collected = []
    while not errors.empty():
        collected.append(errors.get())

    assert not collected, f"concurrent init_local_db failed: {collected}"
    assert all(p.exitcode == 0 for p in procs), (
        f"exit codes {[p.exitcode for p in procs]} — a child died without reporting"
    )


def test_migrations_are_idempotent_when_applied_twice(tmp_path) -> None:
    """The second layer: a step must survive being applied to a database that
    already has its schema.

    Guards the case the ``IF NOT EXISTS`` trigger fix covers — a database
    written by an older SDK, or one where the version was rolled back.
    """
    db = SQLiteHelper(str(tmp_path / "local.db"))
    _run_migrations(db)

    # Force every migration to run again over the finished schema.
    db.execute("PRAGMA user_version = 0")
    _run_migrations(db)  # must not raise

    row = db.fetchone("PRAGMA user_version")
    assert row is not None and row["user_version"] > 0


def test_exclusive_holds_the_write_lock_for_the_whole_block(tmp_path) -> None:
    """A second connection cannot write while the block is open.

    This is the property the migration fix rests on: the version check and the
    schema change are one step as far as any other process is concerned.
    """
    path = str(tmp_path / "local.db")
    db = SQLiteHelper(path)
    db.execute("CREATE TABLE t (a INTEGER)")

    other = sqlite3.connect(path, timeout=0.2)
    try:
        with db.exclusive():
            db.execute("INSERT INTO t VALUES (1)")
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                other.execute("INSERT INTO t VALUES (99)")
                other.commit()
        # Committed on exit, and the intervening write never landed.
        rows = db.fetchall("SELECT a FROM t")
        assert [r["a"] for r in rows] == [1]
    finally:
        other.close()


def test_exclusive_rolls_back_on_error(tmp_path) -> None:
    """A migration that raises must leave no half-applied schema behind."""
    db = SQLiteHelper(str(tmp_path / "local.db"))
    db.execute("CREATE TABLE t (a INTEGER)")

    boom = RuntimeError("migration step failed")
    with pytest.raises(RuntimeError, match="migration step failed"):
        with db.exclusive():
            db.execute("INSERT INTO t VALUES (1)")
            raise boom

    rows = db.fetchall("SELECT a FROM t")
    assert rows == [], "the failed block's write survived the rollback"


def test_nested_exclusive_joins_rather_than_nesting(tmp_path) -> None:
    """SQLite has no nested transactions, so the inner block joins the outer.

    A migration step is free to open one without knowing whether its caller
    already did.
    """
    db = SQLiteHelper(str(tmp_path / "local.db"))
    db.execute("CREATE TABLE t (a INTEGER)")

    with db.exclusive():
        db.execute("INSERT INTO t VALUES (1)")
        with db.exclusive():
            db.execute("INSERT INTO t VALUES (2)")
        # Still inside the outer block — nothing is committed yet.
        db.execute("INSERT INTO t VALUES (3)")

    rows = [r["a"] for r in db.fetchall("SELECT a FROM t ORDER BY a")]
    assert rows == [1, 2, 3]
