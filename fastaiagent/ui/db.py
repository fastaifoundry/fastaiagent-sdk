"""Schema and migrations for the unified local SQLite store.

All local persistence (traces, checkpoints, prompts, eval runs, guardrail events,
and UI view-state) lives in a single file at ``config.local_db_path``
(``./.fastaiagent/local.db`` by default).

Schema version is tracked via ``PRAGMA user_version``. ``init_local_db()``
is idempotent — callers can invoke it on every connect; it only runs the
migrations required to bring the file up to ``CURRENT_SCHEMA_VERSION``.
"""

from __future__ import annotations

from pathlib import Path

from fastaiagent._internal.config import get_config
from fastaiagent._internal.storage import SQLiteHelper

CURRENT_SCHEMA_VERSION = 1


_MIGRATIONS: dict[int, list[str]] = {
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
        "CREATE INDEX IF NOT EXISTS idx_guardrail_events_trace_id"
        " ON guardrail_events(trace_id)",
        "CREATE INDEX IF NOT EXISTS idx_guardrail_events_agent"
        " ON guardrail_events(agent_name)",
        "CREATE INDEX IF NOT EXISTS idx_guardrail_events_rule"
        " ON guardrail_events(guardrail_name)",
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
        for stmt in _MIGRATIONS[version]:
            db.execute(stmt)
        _set_user_version(db, version)


def _get_user_version(db: SQLiteHelper) -> int:
    row = db.fetchone("PRAGMA user_version")
    if not row:
        return 0
    return int(next(iter(row.values())))


def _set_user_version(db: SQLiteHelper, version: int) -> None:
    # PRAGMA does not accept parameter binding — inline an int we control.
    db.execute(f"PRAGMA user_version = {int(version)}")
