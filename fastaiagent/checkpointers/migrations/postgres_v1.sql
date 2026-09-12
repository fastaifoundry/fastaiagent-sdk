-- v1.0 durability schema for Postgres.
-- Applied by ``PostgresCheckpointer.setup()`` once per pool, gated on the
-- ``schema_version`` table so re-runs are no-ops. The schema name is the
-- ``schema=`` constructor argument (default ``fastaiagent``); SQL keeps the
-- name parameterized via the search_path that ``setup()`` sets before
-- running the script.

CREATE SCHEMA IF NOT EXISTS {schema};

CREATE TABLE IF NOT EXISTS {schema}.schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {schema}.checkpoints (
    checkpoint_id        TEXT PRIMARY KEY,
    parent_checkpoint_id TEXT,
    chain_name           TEXT NOT NULL,
    execution_id         TEXT NOT NULL,
    node_id              TEXT NOT NULL,
    node_index           INTEGER,
    status               TEXT NOT NULL DEFAULT 'completed',
    state_snapshot       JSONB NOT NULL,
    node_input           JSONB,
    node_output          JSONB,
    iteration            INTEGER NOT NULL DEFAULT 0,
    iteration_counters   JSONB,
    interrupt_reason     TEXT,
    interrupt_context    JSONB,
    agent_path           TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cp_exec
    ON {schema}.checkpoints (execution_id);

-- WS2 connected-durability outbox flag. DEFAULT TRUE so pre-existing rows are
-- treated as already-synced (connecting does NOT back-push checkpoint history);
-- every new INSERT (put / record_interrupt) sets ``synced = FALSE`` explicitly to
-- become a push candidate. ADD COLUMN IF NOT EXISTS keeps setup() idempotent — a
-- re-run never clobbers un-acked rows.
ALTER TABLE {schema}.checkpoints
    ADD COLUMN IF NOT EXISTS synced BOOLEAN NOT NULL DEFAULT TRUE;

CREATE INDEX IF NOT EXISTS idx_cp_synced
    ON {schema}.checkpoints (synced, created_at);

-- Poison-row quarantine (durability audit D2). A checkpoint the plane's ingest
-- door refuses on PAYLOAD grounds is refused identically every time, and the
-- drain sends the oldest un-acked rows as one batch and stopped at the first
-- failure while leaving them buffered -- so one such row re-sent the same batch
-- forever and stranded every later checkpoint on this store. A quarantined row
-- is marked ``synced = TRUE`` with the reason here, so ``fetch_unsynced`` needs
-- no change (and no sentinel value, which a BOOLEAN could not carry anyway) and
-- ``synced = TRUE AND sync_error IS NOT NULL`` reads as "we gave up on this one"
-- rather than "this reached the plane". Mirrors local.db schema v19.
ALTER TABLE {schema}.checkpoints
    ADD COLUMN IF NOT EXISTS sync_error TEXT;

-- Run-end marker + step classification (durability audit D5). Mirrors local.db
-- schema v20. `run_end` is the value that matters: without it a finished run and
-- one that died right after its last step both leave a `completed` checkpoint as
-- the newest row, so neither the plane nor `aresume` can tell them apart. The
-- plane's wire schema has carried `step_type` since WS2 (capped at 40 chars);
-- the SDK just never sent it. No backfill -- NULL on an existing row is honest.
ALTER TABLE {schema}.checkpoints
    ADD COLUMN IF NOT EXISTS step_type TEXT;

-- Partial index for the /approvals + Failed Executions pages — most rows
-- are 'completed' and don't need to be scanned.
CREATE INDEX IF NOT EXISTS idx_cp_status_problem
    ON {schema}.checkpoints (status)
    WHERE status IN ('failed', 'interrupted');

CREATE TABLE IF NOT EXISTS {schema}.idempotency_cache (
    execution_id TEXT NOT NULL,
    function_key TEXT NOT NULL,
    result       JSONB NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (execution_id, function_key)
);

CREATE TABLE IF NOT EXISTS {schema}.pending_interrupts (
    execution_id TEXT PRIMARY KEY,
    chain_name   TEXT NOT NULL,
    node_id      TEXT NOT NULL,
    reason       TEXT NOT NULL,
    context      JSONB NOT NULL,
    agent_path   TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO {schema}.schema_version (version)
    VALUES (1)
    ON CONFLICT DO NOTHING;
