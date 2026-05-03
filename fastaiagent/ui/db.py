"""Schema and migrations for the unified local SQLite store.

All local persistence (traces, checkpoints, prompts, eval runs, guardrail events,
and UI view-state) lives in a single file at ``config.local_db_path``
(``./.fastaiagent/local.db`` by default).

Schema version is tracked via ``PRAGMA user_version``. ``init_local_db()``
is idempotent — callers can invoke it on every connect; it only runs the
migrations required to bring the file up to ``CURRENT_SCHEMA_VERSION``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastaiagent._internal.config import get_config
from fastaiagent._internal.storage import SQLiteHelper

CURRENT_SCHEMA_VERSION = 5

# A migration step is either a SQL string or a callable that takes the
# ``SQLiteHelper`` and runs whatever logic it needs (e.g., gated
# ``ALTER TABLE`` for SQLite versions before 3.35).
_Step = str | Callable[[SQLiteHelper], None]


def _add_column_if_missing(db: SQLiteHelper, table: str, column: str, ddl: str) -> None:
    """``ALTER TABLE table ADD COLUMN column ddl`` if the column does not exist.

    SQLite < 3.35 lacks ``ADD COLUMN IF NOT EXISTS``; check ``PRAGMA
    table_info`` first.
    """
    rows = db.fetchall(f"PRAGMA table_info({table})")
    existing = {r["name"] for r in rows}
    if column in existing:
        return
    db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _v2_add_checkpoint_columns(db: SQLiteHelper) -> None:
    _add_column_if_missing(db, "checkpoints", "checkpoint_id", "TEXT")
    _add_column_if_missing(db, "checkpoints", "parent_checkpoint_id", "TEXT")
    _add_column_if_missing(db, "checkpoints", "interrupt_reason", "TEXT")
    _add_column_if_missing(db, "checkpoints", "interrupt_context", "TEXT DEFAULT '{}'")
    _add_column_if_missing(db, "checkpoints", "agent_path", "TEXT")


# v4 — project scoping. Stamp every UI-visible record with project_id so the
# same DB can host multiple projects.
_PROJECT_TABLES = (
    "spans",
    "checkpoints",
    "pending_interrupts",
    "idempotency_cache",
    "trace_attachments",
    "prompts",
    "prompt_versions",
    "eval_runs",
    "eval_cases",
    "guardrail_events",
)


def _v4_add_project_id_columns(db: SQLiteHelper) -> None:
    """Add ``project_id TEXT NOT NULL DEFAULT ''`` to every project-scoped
    table. SQLite < 3.35 doesn't support ``ADD COLUMN IF NOT EXISTS``, so
    we check ``PRAGMA table_info`` per table.
    """
    for table in _PROJECT_TABLES:
        # Some installs may not have every table (e.g. legacy DB without
        # idempotency_cache). Skip rather than crash.
        rows = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        if not rows:
            continue
        _add_column_if_missing(
            db, table, "project_id", "TEXT NOT NULL DEFAULT ''"
        )


def _v4_backfill_project_id(db: SQLiteHelper) -> None:
    """Backfill empty ``project_id`` rows with the current project's id.

    Runs once per database; idempotent because it only updates rows where
    ``project_id = ''``.
    """
    from fastaiagent._internal.project import safe_get_project_id

    pid = safe_get_project_id()
    if not pid:
        return
    for table in _PROJECT_TABLES:
        rows = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        if not rows:
            continue
        db.execute(
            f"UPDATE {table} SET project_id = ? WHERE project_id = ''",
            (pid,),
        )


def _v5_add_false_positive_columns(db: SQLiteHelper) -> None:
    """Annotate guardrail events as false positives.

    Two columns:
      * ``false_positive`` — 0/1 flag set by the developer via the UI's
        "Mark as false positive" button.
      * ``false_positive_at`` — ISO timestamp of the most recent toggle.

    The columns are nullable / default 0 so historical events stay valid
    without a backfill. Toggling and untoggling both flow through the
    same PATCH endpoint, which keeps the timestamp current.
    """
    rows = db.fetchall(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='guardrail_events'"
    )
    if not rows:
        return
    _add_column_if_missing(
        db, "guardrail_events", "false_positive", "INTEGER DEFAULT 0"
    )
    _add_column_if_missing(db, "guardrail_events", "false_positive_at", "TEXT")


def _v4_create_project_indexes(db: SQLiteHelper) -> None:
    """Add per-project hot-path indexes on tables that exist.

    Legacy fixtures (checkpoint-only DBs) don't have a ``spans``
    table; ``IF NOT EXISTS`` on the index doesn't protect against the
    underlying table being missing, so we gate the CREATE INDEX on a
    sqlite_master probe.
    """
    for table, sql in [
        (
            "spans",
            "CREATE INDEX IF NOT EXISTS idx_spans_project "
            "ON spans(project_id, start_time DESC)",
        ),
        (
            "checkpoints",
            "CREATE INDEX IF NOT EXISTS idx_checkpoints_project "
            "ON checkpoints(project_id, execution_id)",
        ),
    ]:
        rows = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        if rows:
            db.execute(sql)


_MIGRATIONS: dict[int, list[_Step]] = {
    1: [
        # Trace spans (moved from traces.db).
        """CREATE TABLE IF NOT EXISTS spans (
            span_id        TEXT PRIMARY KEY,
            trace_id       TEXT NOT NULL,
            parent_span_id TEXT,
            name           TEXT,
            start_time     TEXT,
            end_time       TEXT,
            status         TEXT DEFAULT 'OK',
            attributes     TEXT DEFAULT '{}',
            events         TEXT DEFAULT '[]'
        )""",
        "CREATE INDEX IF NOT EXISTS idx_spans_trace_id   ON spans (trace_id)",
        "CREATE INDEX IF NOT EXISTS idx_spans_start_time ON spans (start_time)",
        # Chain checkpoints (moved from checkpoints.db).
        """CREATE TABLE IF NOT EXISTS checkpoints (
            id                 TEXT PRIMARY KEY,
            chain_name         TEXT NOT NULL,
            execution_id       TEXT NOT NULL,
            node_id            TEXT NOT NULL,
            node_index         INTEGER,
            status             TEXT DEFAULT 'completed',
            state_snapshot     TEXT DEFAULT '{}',
            node_input         TEXT DEFAULT '{}',
            node_output        TEXT DEFAULT '{}',
            iteration          INTEGER DEFAULT 0,
            iteration_counters TEXT DEFAULT '{}',
            created_at         TEXT
        )""",
        "CREATE INDEX IF NOT EXISTS idx_cp_exec ON checkpoints (execution_id)",
        # Prompt registry (replaces YAML).
        """CREATE TABLE IF NOT EXISTS prompts (
            slug           TEXT PRIMARY KEY,
            description    TEXT,
            tags           TEXT,
            latest_version TEXT,
            created_at     TEXT,
            updated_at     TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS prompt_versions (
            slug       TEXT NOT NULL,
            version    TEXT NOT NULL,
            template   TEXT,
            variables  TEXT,
            fragments  TEXT,
            metadata   TEXT,
            created_at TEXT,
            created_by TEXT,
            PRIMARY KEY (slug, version),
            FOREIGN KEY (slug) REFERENCES prompts(slug)
        )""",
        """CREATE TABLE IF NOT EXISTS prompt_aliases (
            slug    TEXT NOT NULL,
            alias   TEXT NOT NULL,
            version TEXT NOT NULL,
            PRIMARY KEY (slug, alias),
            FOREIGN KEY (slug) REFERENCES prompts(slug)
        )""",
        """CREATE TABLE IF NOT EXISTS prompt_fragments (
            name       TEXT PRIMARY KEY,
            content    TEXT,
            created_at TEXT,
            updated_at TEXT
        )""",
        # Eval runs + cases.
        """CREATE TABLE IF NOT EXISTS eval_runs (
            run_id        TEXT PRIMARY KEY,
            run_name      TEXT,
            dataset_name  TEXT,
            agent_name    TEXT,
            agent_version TEXT,
            scorers       TEXT,
            started_at    TEXT,
            finished_at   TEXT,
            pass_count    INTEGER,
            fail_count    INTEGER,
            pass_rate     REAL,
            metadata      TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS eval_cases (
            case_id         TEXT PRIMARY KEY,
            run_id          TEXT NOT NULL,
            ordinal         INTEGER,
            input           TEXT,
            expected_output TEXT,
            actual_output   TEXT,
            trace_id        TEXT,
            per_scorer      TEXT,
            FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_eval_cases_run_id ON eval_cases(run_id)",
        # Guardrail events.
        """CREATE TABLE IF NOT EXISTS guardrail_events (
            event_id       TEXT PRIMARY KEY,
            trace_id       TEXT,
            span_id        TEXT,
            guardrail_name TEXT,
            guardrail_type TEXT,
            position       TEXT,
            outcome        TEXT,
            score          REAL,
            message        TEXT,
            agent_name     TEXT,
            timestamp      TEXT,
            metadata       TEXT
        )""",
        "CREATE INDEX IF NOT EXISTS idx_guardrail_events_trace_id ON guardrail_events(trace_id)",
        "CREATE INDEX IF NOT EXISTS idx_guardrail_events_agent ON guardrail_events(agent_name)",
        "CREATE INDEX IF NOT EXISTS idx_guardrail_events_rule ON guardrail_events(guardrail_name)",
        # UI view-state.
        """CREATE TABLE IF NOT EXISTS trace_notes (
            trace_id   TEXT PRIMARY KEY,
            note       TEXT,
            updated_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS trace_favorites (
            trace_id   TEXT PRIMARY KEY,
            created_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS saved_filters (
            id         TEXT PRIMARY KEY,
            name       TEXT,
            filters    TEXT,
            created_at TEXT
        )""",
    ],
    2: [
        # v1.0 durability — extend `checkpoints` and add new tables.
        _v2_add_checkpoint_columns,
        # Backfill checkpoint_id for existing rows so older executions stay
        # resumable through the new SQLiteCheckpointer.
        "UPDATE checkpoints SET checkpoint_id = id WHERE checkpoint_id IS NULL",
        "CREATE INDEX IF NOT EXISTS idx_cp_checkpoint_id ON checkpoints(checkpoint_id)",
        # Partial index for the failure / interrupt list views (Approvals,
        # Failed Executions). Cheap to maintain and dramatically narrows the
        # scan when most rows are 'completed'.
        "CREATE INDEX IF NOT EXISTS idx_cp_status_problem"
        " ON checkpoints(status) WHERE status IN ('failed','interrupted')",
        # Idempotency cache for the @idempotent decorator (Phase 3).
        """CREATE TABLE IF NOT EXISTS idempotency_cache (
            execution_id TEXT NOT NULL,
            function_key TEXT NOT NULL,
            result       TEXT NOT NULL,
            created_at   TEXT NOT NULL,
            PRIMARY KEY (execution_id, function_key)
        )""",
        # Pending interrupts for the /approvals UI and resume coordination.
        """CREATE TABLE IF NOT EXISTS pending_interrupts (
            execution_id TEXT NOT NULL PRIMARY KEY,
            chain_name   TEXT NOT NULL,
            node_id      TEXT NOT NULL,
            reason       TEXT NOT NULL,
            context      TEXT NOT NULL,
            agent_path   TEXT,
            created_at   TEXT NOT NULL
        )""",
    ],
    3: [
        # Multimodal attachments. Span ``attributes`` JSON only stores
        # metadata + thumbnails; full bytes live here so the trace DB
        # doesn't balloon. ``thumbnail`` is always populated; ``full_data``
        # only when ``fa.config.trace_full_images=True``.
        """CREATE TABLE IF NOT EXISTS trace_attachments (
            attachment_id  TEXT PRIMARY KEY,
            trace_id       TEXT NOT NULL,
            span_id        TEXT NOT NULL,
            media_type     TEXT NOT NULL,
            size_bytes     INTEGER NOT NULL,
            thumbnail      BLOB,
            full_data      BLOB,
            metadata_json  TEXT DEFAULT '{}',
            created_at     TEXT NOT NULL
        )""",
        """CREATE INDEX IF NOT EXISTS idx_trace_attachments_span
            ON trace_attachments(trace_id, span_id)""",
    ],
    4: [
        # Project scoping. Every UI-visible record gets a project_id stamp
        # so the same Postgres can host multiple projects without
        # cross-contamination. SQLite gets the column too — redundant for
        # one-DB-per-project use but lets data carry its project across
        # SQLite → Postgres migration.
        _v4_add_project_id_columns,
        _v4_backfill_project_id,
        # Per-project hot-path indexes. Gated on the underlying table
        # existing — legacy DBs (e.g. checkpoint-only fixtures) might
        # not have a ``spans`` table at all, and ``IF NOT EXISTS`` on
        # the index doesn't protect against the table being missing.
        _v4_create_project_indexes,
    ],
    5: [
        # Sprint 2 — Guardrail Event Detail. The "Mark as false positive"
        # button records its annotation directly on the existing event
        # row (rather than a side table) so historical data remains
        # editable in place and so the list endpoint can filter on
        # ``false_positive`` without a join.
        _v5_add_false_positive_columns,
    ],
}


def init_local_db(db_path: str | Path | None = None) -> SQLiteHelper:
    """Open ``local.db`` and run any outstanding migrations.

    Returns a ready-to-use ``SQLiteHelper`` pointed at ``db_path`` (or the
    configured ``local_db_path``). Safe to call multiple times — migrations are
    idempotent and gated on ``PRAGMA user_version``.
    """
    resolved = str(db_path) if db_path is not None else get_config().local_db_path
    db = SQLiteHelper(resolved)
    _run_migrations(db)
    return db


def _run_migrations(db: SQLiteHelper) -> None:
    current = _get_user_version(db)
    for version in sorted(_MIGRATIONS):
        if version <= current:
            continue
        for step in _MIGRATIONS[version]:
            if isinstance(step, str):
                db.execute(step)
            else:
                step(db)
        _set_user_version(db, version)


def _get_user_version(db: SQLiteHelper) -> int:
    row = db.fetchone("PRAGMA user_version")
    if not row:
        return 0
    return int(next(iter(row.values())))


def _set_user_version(db: SQLiteHelper, version: int) -> None:
    # PRAGMA does not accept parameter binding — inline an int we control.
    db.execute(f"PRAGMA user_version = {int(version)}")
