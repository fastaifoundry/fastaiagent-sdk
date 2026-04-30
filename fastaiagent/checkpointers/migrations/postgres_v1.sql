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
