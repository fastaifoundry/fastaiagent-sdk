"""Marking spans exported must not reindex them (1.91.0, X19; schema v27).

``spans_fts_au`` fired on every update of ``spans``, including the exporter's
``UPDATE spans SET synced = 1``. Each firing deletes from ``span_fts`` by the
UNINDEXED ``span_id`` — a scan of the whole index. On a real 380k-span
``local.db`` that was 0.15 s per span; abandoning a backlog of 179k stale spans
(``enforce_buffer_bound``) ran for about 7.5 hours, holding span export and
``fa.disconnect()`` the whole time. Found when the connected visibility gate hung.

Since v27 the trigger fires on ``trace_id`` / ``name`` / ``attributes`` only —
what the index is built from — so a synced flip leaves the index alone and an
attributes change is still searchable.

Real SQLite files and the real ``TraceStore`` (no mocks). The reindex is observed
directly: a reindexed row gets a new FTS ``rowid``.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.trace.storage import TraceStore
from fastaiagent.ui.db import CURRENT_SCHEMA_VERSION, init_local_db

OLD = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()


def _seed(db: SQLiteHelper, n: int, *, project: str = "p1", start: str = OLD) -> list[str]:
    ids = [f"span-{i}" for i in range(n)]
    for span_id in ids:
        db.execute(
            "INSERT INTO spans (span_id, trace_id, name, start_time, attributes, project_id,"
            " synced) VALUES (?, ?, ?, ?, ?, ?, 0)",
            (
                span_id,
                f"t-{span_id}",
                "llm.openai",
                start,
                json.dumps({"gen_ai.prompt": f"refund policy {span_id}"}),
                project,
            ),
        )
    return ids


def _fts_rowids(db: SQLiteHelper) -> dict[str, int]:
    return {r["span_id"]: r["rowid"] for r in db.fetchall("SELECT rowid, span_id FROM span_fts")}


def test_marking_spans_synced_leaves_the_index_alone(tmp_path) -> None:
    path = tmp_path / "local.db"
    db = init_local_db(path)
    ids = _seed(db, 50)
    before = _fts_rowids(db)
    db.close()

    store = TraceStore(db_path=str(path))
    store.mark_synced(ids)

    after = _fts_rowids(store._db)
    assert after == before, "a synced flip reindexed spans"
    assert store.count_unsynced() == 0


def test_abandoning_a_stale_backlog_leaves_the_index_alone(tmp_path) -> None:
    """The exact path that hung: ``enforce_buffer_bound``'s age bound."""
    path = tmp_path / "local.db"
    db = init_local_db(path)
    _seed(db, 200)
    before = _fts_rowids(db)
    db.close()

    store = TraceStore(db_path=str(path))
    abandoned = store.enforce_buffer_bound(max_unsynced=10_000, max_age_days=7, project_id="p1")

    assert abandoned == 200
    assert _fts_rowids(store._db) == before


def test_an_attributes_change_is_still_searchable(tmp_path) -> None:
    db = init_local_db(tmp_path / "local.db")
    (span_id,) = _seed(db, 1)
    db.execute(
        "UPDATE spans SET attributes = ? WHERE span_id = ?",
        (json.dumps({"gen_ai.prompt": "warranty claim"}), span_id),
    )
    hits = db.fetchall("SELECT span_id FROM span_fts WHERE span_fts MATCH 'warranty'")
    assert [h["span_id"] for h in hits] == [span_id]
    assert not db.fetchall("SELECT span_id FROM span_fts WHERE span_fts MATCH 'refund'")


def test_a_v26_database_is_upgraded(tmp_path) -> None:
    """An existing local.db carries the old trigger; opening it upgrades it."""
    path = tmp_path / "local.db"
    db = init_local_db(path)
    # Put the pre-v27 trigger back and mark the file v26, as a 1.90.0 DB is.
    db.execute("DROP TRIGGER spans_fts_au")
    db.execute(
        """CREATE TRIGGER spans_fts_au AFTER UPDATE ON spans BEGIN
               DELETE FROM span_fts WHERE span_id = old.span_id;
               INSERT INTO span_fts(trace_id, span_id, name, input_text, output_text)
               VALUES (new.trace_id, new.span_id, new.name, '', '');
           END"""
    )
    db.execute("PRAGMA user_version = 26")
    ids = _seed(db, 5)
    db.close()

    upgraded = init_local_db(path)
    version = next(iter(upgraded.fetchone("PRAGMA user_version").values()))
    assert version == CURRENT_SCHEMA_VERSION >= 27
    before = _fts_rowids(upgraded)
    upgraded.execute(
        f"UPDATE spans SET synced = 1 WHERE span_id IN ({','.join('?' * len(ids))})", tuple(ids)
    )
    assert _fts_rowids(upgraded) == before
