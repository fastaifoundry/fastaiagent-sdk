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

CURRENT_SCHEMA_VERSION = 11

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
    "external_agents",
    "external_agent_attachments",
    "sim_runs",
    "sim_cases",
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
        _add_column_if_missing(db, table, "project_id", "TEXT NOT NULL DEFAULT ''")


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


def _v6_add_span_fts(db: SQLiteHelper) -> None:
    """Sprint 3 — full-text search across span LLM prompts/responses.

    Creates a FTS5 virtual table mirroring two extracted JSON fields per
    span (``gen_ai.prompt`` and ``gen_ai.response.text``), plus three
    triggers that keep the FTS table in sync as spans are
    inserted/updated/deleted. Existing rows are backfilled in a single
    ``INSERT … SELECT`` so the migration is fast even on million-span
    DBs.

    Skipped when:
      * The ``spans`` table doesn't exist (legacy checkpoint-only DBs).
      * The SQLite build was compiled without FTS5 — ``CREATE VIRTUAL
        TABLE … USING fts5`` raises ``OperationalError`` and we treat
        that as "search degrades to LIKE" rather than failing the whole
        migration.

    Postgres parity: the UI's read tables (spans/eval_runs/etc.) are
    SQLite-only in this repo. The Postgres deployment is currently
    checkpointer-only (see ``checkpointers/migrations/postgres_v1.sql``).
    When the read side moves to Postgres, the equivalent index is::

        CREATE INDEX idx_spans_attributes_fts
            ON fastaiagent.spans
            USING gin (to_tsvector('english', attributes::text));

    Add it to the Postgres migration sibling at that point.
    """
    rows = db.fetchall("SELECT name FROM sqlite_master WHERE type='table' AND name='spans'")
    if not rows:
        return

    try:
        db.execute(
            """CREATE VIRTUAL TABLE IF NOT EXISTS span_fts USING fts5(
                trace_id,
                span_id UNINDEXED,
                name,
                input_text,
                output_text,
                tokenize = 'unicode61'
            )"""
        )
    except Exception:
        # SQLite without FTS5 — leave the LIKE-fallback path in
        # ``list_traces`` doing its job.
        return

    db.execute(
        """CREATE TRIGGER IF NOT EXISTS spans_fts_ai
           AFTER INSERT ON spans BEGIN
               INSERT INTO span_fts(trace_id, span_id, name, input_text, output_text)
               VALUES (
                   new.trace_id,
                   new.span_id,
                   new.name,
                   COALESCE(json_extract(new.attributes, '$."gen_ai.prompt"'),
                            json_extract(new.attributes, '$."fastaiagent.gen_ai.prompt"'),
                            ''),
                   COALESCE(json_extract(new.attributes, '$."gen_ai.response.text"'),
                            json_extract(new.attributes, '$."gen_ai.completion"'),
                            json_extract(new.attributes, '$."fastaiagent.gen_ai.response.text"'),
                            '')
               );
           END"""
    )
    db.execute(
        """CREATE TRIGGER IF NOT EXISTS spans_fts_ad
           AFTER DELETE ON spans BEGIN
               DELETE FROM span_fts WHERE span_id = old.span_id;
           END"""
    )
    db.execute(
        """CREATE TRIGGER IF NOT EXISTS spans_fts_au
           AFTER UPDATE ON spans BEGIN
               DELETE FROM span_fts WHERE span_id = old.span_id;
               INSERT INTO span_fts(trace_id, span_id, name, input_text, output_text)
               VALUES (
                   new.trace_id,
                   new.span_id,
                   new.name,
                   COALESCE(json_extract(new.attributes, '$."gen_ai.prompt"'),
                            json_extract(new.attributes, '$."fastaiagent.gen_ai.prompt"'),
                            ''),
                   COALESCE(json_extract(new.attributes, '$."gen_ai.response.text"'),
                            json_extract(new.attributes, '$."gen_ai.completion"'),
                            json_extract(new.attributes, '$."fastaiagent.gen_ai.response.text"'),
                            '')
               );
           END"""
    )

    # Bulk backfill of any pre-existing rows. Safe to re-run because we
    # first wipe span_fts — the rebuild is cheap and avoids duplicates
    # when an admin re-runs init_local_db on a populated DB after an
    # external schema reset.
    existing = db.fetchone("SELECT COUNT(*) AS n FROM span_fts")
    if existing and (existing.get("n") or 0) == 0:
        db.execute(
            """INSERT INTO span_fts(trace_id, span_id, name, input_text, output_text)
               SELECT
                   trace_id,
                   span_id,
                   name,
                   COALESCE(json_extract(attributes, '$."gen_ai.prompt"'),
                            json_extract(attributes, '$."fastaiagent.gen_ai.prompt"'),
                            ''),
                   COALESCE(json_extract(attributes, '$."gen_ai.response.text"'),
                            json_extract(attributes, '$."gen_ai.completion"'),
                            json_extract(attributes, '$."fastaiagent.gen_ai.response.text"'),
                            '')
               FROM spans"""
        )


# Full-text search content extraction. These are the attribute keys whose
# values hold the text a user actually searches for — "what the agent sent and
# received". Ordered by preference; ``COALESCE`` picks the first present per
# span. Native fastaiagent spans store the agent turn under
# ``agent.input``/``agent.output`` and the raw LLM exchange under
# ``gen_ai.request.messages``; ``gen_ai.prompt``/``gen_ai.response.text`` are the
# canonical keys produced by foreign-OTel normalization; tool spans expose
# ``tool.args``/``tool.result``. Indexing all of them makes the search box match
# inputs and outputs, not just span names (see ``_v10_widen_span_fts``).
_FTS_INPUT_KEYS = (
    "gen_ai.prompt",
    "fastaiagent.gen_ai.prompt",
    "agent.input",
    "gen_ai.request.messages",
    "tool.args",
)
_FTS_OUTPUT_KEYS = (
    "gen_ai.response.text",
    "gen_ai.completion",
    "fastaiagent.gen_ai.response.text",
    "agent.output",
    "gen_ai.response.tool_calls",
    "tool.result",
)


def _coalesce_json(attrs_col: str, keys: tuple[str, ...]) -> str:
    """SQL expression pulling the first present attribute value from ``keys``
    out of the JSON column ``attrs_col``, falling back to ``''``."""
    extracts = ", ".join(f"json_extract({attrs_col}, '$.\"{key}\"')" for key in keys)
    return f"COALESCE({extracts}, '')"


def _v10_widen_span_fts(db: SQLiteHelper) -> None:
    """Broaden full-text search to cover agent/tool inputs & outputs.

    The v6 FTS triggers indexed only ``gen_ai.prompt`` / ``gen_ai.response.text``
    — keys native fastaiagent spans never populate — so ``/api/traces?q=...``
    matched span names only. This rebuilds the three sync triggers *and* the
    existing index content over the wider key set in ``_FTS_INPUT_KEYS`` /
    ``_FTS_OUTPUT_KEYS`` so a search like "refund policy" matches the trace whose
    agent input/output or LLM messages contained the phrase.

    No-op when ``span_fts`` is absent (FTS5 not compiled in, or v6 skipped) — the
    ``list_traces`` LIKE-on-JSON fallback already covers that build.
    """
    rows = db.fetchall("SELECT name FROM sqlite_master WHERE type='table' AND name='span_fts'")
    if not rows:
        return

    in_new = _coalesce_json("new.attributes", _FTS_INPUT_KEYS)
    out_new = _coalesce_json("new.attributes", _FTS_OUTPUT_KEYS)

    for trig in ("spans_fts_ai", "spans_fts_au", "spans_fts_ad"):
        db.execute(f"DROP TRIGGER IF EXISTS {trig}")

    db.execute(
        f"""CREATE TRIGGER spans_fts_ai
           AFTER INSERT ON spans BEGIN
               INSERT INTO span_fts(trace_id, span_id, name, input_text, output_text)
               VALUES (new.trace_id, new.span_id, new.name, {in_new}, {out_new});
           END"""
    )
    db.execute(
        """CREATE TRIGGER spans_fts_ad
           AFTER DELETE ON spans BEGIN
               DELETE FROM span_fts WHERE span_id = old.span_id;
           END"""
    )
    db.execute(
        f"""CREATE TRIGGER spans_fts_au
           AFTER UPDATE ON spans BEGIN
               DELETE FROM span_fts WHERE span_id = old.span_id;
               INSERT INTO span_fts(trace_id, span_id, name, input_text, output_text)
               VALUES (new.trace_id, new.span_id, new.name, {in_new}, {out_new});
           END"""
    )

    # Rebuild content for all existing rows under the wider key set.
    in_col = _coalesce_json("attributes", _FTS_INPUT_KEYS)
    out_col = _coalesce_json("attributes", _FTS_OUTPUT_KEYS)
    db.execute("DELETE FROM span_fts")
    db.execute(
        f"""INSERT INTO span_fts(trace_id, span_id, name, input_text, output_text)
           SELECT trace_id, span_id, name, {in_col}, {out_col} FROM spans"""
    )


def _v6_add_saved_filters_project(db: SQLiteHelper) -> None:
    """Sprint 3 — make the v1 saved_filters table project-scoped.

    The table existed since v1 but was never used by any code path.
    Sprint 3 wires it up via ``/api/filter-presets``; reusing the
    existing schema avoids a parallel ``filter_presets`` table.
    """
    rows = db.fetchall("SELECT name FROM sqlite_master WHERE type='table' AND name='saved_filters'")
    if not rows:
        return
    _add_column_if_missing(db, "saved_filters", "project_id", "TEXT NOT NULL DEFAULT ''")
    db.execute("CREATE INDEX IF NOT EXISTS idx_saved_filters_project ON saved_filters(project_id)")


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
        "SELECT name FROM sqlite_master WHERE type='table' AND name='guardrail_events'"
    )
    if not rows:
        return
    _add_column_if_missing(db, "guardrail_events", "false_positive", "INTEGER DEFAULT 0")
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
            "CREATE INDEX IF NOT EXISTS idx_spans_project ON spans(project_id, start_time DESC)",
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


def _v11_add_span_synced(db: SQLiteHelper) -> None:
    """Durable platform-export buffer flag on ``spans``.

    ``PlatformSpanExporter`` marks a span ``synced=1`` only after a confirmed
    2xx push to ``/public/v1/traces/ingest``; until then the row is a buffered
    re-send candidate, drained at the top of each ``export()``.

    Existing rows are backfilled to ``synced=1`` so upgrading does NOT
    retroactively back-push a user's entire local trace history on the first
    ``export()`` after the upgrade — only spans created afterwards become push
    candidates. Manual backfill of historical traces stays available via
    :meth:`fastaiagent.trace.storage.TraceData.publish`.

    Gated on the ``spans`` table existing (legacy checkpoint-only DBs don't have
    it). SQLite has no native boolean; ``INTEGER`` 0/1 matches the existing
    ``false_positive`` convention.
    """
    rows = db.fetchall("SELECT name FROM sqlite_master WHERE type='table' AND name='spans'")
    if not rows:
        return
    _add_column_if_missing(db, "spans", "synced", "INTEGER NOT NULL DEFAULT 0")
    # Backfill pre-existing rows as already-handled so the upgrade is silent.
    db.execute("UPDATE spans SET synced = 1 WHERE synced = 0")
    db.execute("CREATE INDEX IF NOT EXISTS idx_spans_synced ON spans(synced, start_time)")


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
    6: [
        # Sprint 3 — Richer Trace Filtering.
        #
        # 1. ``span_fts`` (FTS5 virtual table) + sync triggers + bulk
        #    backfill so ``/api/traces?q=...`` matches against extracted
        #    LLM prompt/response text instead of LIKE on JSON blobs.
        # 2. Project-scope the v1 ``saved_filters`` table so the new
        #    ``/api/filter-presets`` endpoints can be project-isolated
        #    without a parallel table.
        _v6_add_span_fts,
        _v6_add_saved_filters_project,
    ],
    7: [
        # Sprint 4 — Universal harness external-agent registry.
        #
        # ``register_agent()`` and the harness auto-attachment helpers
        # write into these two tables. The ``/api/agents/{name}/dependencies``
        # endpoint merges them with the in-memory ``ctx.runners`` lookup
        # so external agents (LangGraph, CrewAI, PydanticAI) show up in
        # the dependency-graph UI alongside native runners.
        """CREATE TABLE IF NOT EXISTS external_agents (
            name           TEXT PRIMARY KEY,
            framework      TEXT NOT NULL,
            model          TEXT,
            provider       TEXT,
            system_prompt  TEXT,
            topology_json  TEXT DEFAULT '{}',
            metadata_json  TEXT DEFAULT '{}',
            created_at     TEXT NOT NULL,
            updated_at     TEXT NOT NULL,
            project_id     TEXT NOT NULL DEFAULT ''
        )""",
        """CREATE TABLE IF NOT EXISTS external_agent_attachments (
            attachment_id  TEXT PRIMARY KEY,
            agent_name     TEXT NOT NULL,
            kind           TEXT NOT NULL,
            ref_name       TEXT NOT NULL,
            position       TEXT,
            version        TEXT,
            metadata_json  TEXT DEFAULT '{}',
            created_at     TEXT NOT NULL,
            project_id     TEXT NOT NULL DEFAULT ''
        )""",
        "CREATE INDEX IF NOT EXISTS idx_ext_agents_framework ON external_agents(framework)",
        "CREATE INDEX IF NOT EXISTS idx_ext_attach_agent ON external_agent_attachments(agent_name)",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_ext_attach "
        "ON external_agent_attachments(agent_name, kind, ref_name, position)",
    ],
    8: [
        # Trace Learning Loop — durable per-user/per-project/per-agent facts
        # extracted offline from completed traces by ``fastaiagent learn``.
        # Re-injected into future runs via ``PersistentFactBlock``. The
        # ``superseded_by`` chain encodes conflict-resolution by recency
        # (newest fact wins; older row marks itself superseded), so we keep
        # the audit trail rather than deleting.
        """CREATE TABLE IF NOT EXISTS learned_memory (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            scope           TEXT NOT NULL,
            scope_id        TEXT NOT NULL DEFAULT '',
            fact            TEXT NOT NULL,
            source_trace_id TEXT,
            confidence      REAL DEFAULT 1.0,
            created_at      REAL NOT NULL,
            superseded_by   INTEGER,
            project_id      TEXT NOT NULL DEFAULT '',
            UNIQUE(scope, scope_id, fact, project_id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_learned_memory_scope "
        "ON learned_memory(scope, scope_id, project_id)",
        "CREATE INDEX IF NOT EXISTS idx_learned_memory_created ON learned_memory(created_at)",
    ],
    9: [
        # Agent simulation runs + per-scenario cases. Mirrors the eval_runs /
        # eval_cases pair: ``simulate()`` writes one ``sim_runs`` row and one
        # ``sim_cases`` row per scenario. The transcript + per-criterion
        # verdicts are stored as JSON on the case row (like eval_cases.per_scorer)
        # so the surface stays two tables and reuses the evals rendering pattern.
        """CREATE TABLE IF NOT EXISTS sim_runs (
            run_id         TEXT PRIMARY KEY,
            run_name       TEXT,
            agent_name     TEXT,
            scenario_count INTEGER,
            pass_count     INTEGER,
            fail_count     INTEGER,
            pass_rate      REAL,
            started_at     TEXT,
            finished_at    TEXT,
            metadata       TEXT,
            project_id     TEXT NOT NULL DEFAULT ''
        )""",
        """CREATE TABLE IF NOT EXISTS sim_cases (
            case_id       TEXT PRIMARY KEY,
            run_id        TEXT NOT NULL,
            ordinal       INTEGER,
            scenario_name TEXT,
            passed        INTEGER,
            criteria      TEXT,          -- JSON: {"success": [...], "failure": [...]}
            per_criterion TEXT,          -- JSON: [{"criterion","kind","passed","reason"}]
            transcript    TEXT,          -- JSON: [{"turn_index","role","content","trace_id"}]
            trace_id      TEXT,          -- root simulation trace
            project_id    TEXT NOT NULL DEFAULT '',
            FOREIGN KEY (run_id) REFERENCES sim_runs(run_id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_sim_cases_run ON sim_cases(run_id, project_id)",
    ],
    10: [
        # Widen full-text search. The v6 triggers indexed only
        # gen_ai.prompt/response.text — keys native spans don't populate — so the
        # search box matched span names only. Rebuild the FTS triggers + content
        # over agent.input/output, gen_ai.request.messages, and tool.args/result
        # so /api/traces?q=... matches real inputs and outputs.
        _v10_widen_span_fts,
    ],
    11: [
        # Durable platform-export buffer. Add a ``synced`` flag to ``spans`` so
        # PlatformSpanExporter can buffer un-acked spans across an outage and
        # re-drain them on the next successful export. Existing rows are
        # backfilled to synced=1 so the upgrade doesn't back-push history.
        _v11_add_span_synced,
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
