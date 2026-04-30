# Checkpointers

The `Checkpointer` Protocol is the storage surface every backend must
satisfy. v1.0 ships two implementations: SQLite (local development /
single-process production) and Postgres (multi-process / distributed
production). Both pass the same parameterized integration suite —
[`tests/integration/test_postgres_checkpointer.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/tests/integration/test_postgres_checkpointer.py)
runs the full protocol contract against each backend on every CI build.

## Picking a backend

| You want… | Use |
|---|---|
| Local development. One Python process. Zero ops. | `SQLiteCheckpointer` |
| Production behind one container. SQLite on a persistent volume. | `SQLiteCheckpointer` |
| Multi-replica HTTP service where any worker can resume any execution | `PostgresCheckpointer` |
| Concurrent resumers on the same execution_id | `PostgresCheckpointer` |
| A real-time stress test of the atomic-claim contract | `PostgresCheckpointer` (its `DELETE … RETURNING *` is the cleanest) |

Both backends respect the same atomic-claim contract: of N concurrent
resumers, exactly one wins and the rest see `AlreadyResumed`. SQLite's
single-writer model makes the race window small in practice; Postgres
MVCC makes it formally bulletproof.

## SQLite

The default. No configuration needed.

```python
from fastaiagent import Chain, SQLiteCheckpointer

chain = Chain(
    "my-flow",
    checkpointer=SQLiteCheckpointer(),  # writes to ./.fastaiagent/local.db
)
```

Custom path:

```python
SQLiteCheckpointer(db_path="/var/lib/fastaiagent/state.db")
```

Or set the `FASTAIAGENT_LOCAL_DB` environment variable.

The checkpoint store **shares** `local.db` with traces, prompts, eval
runs, and other local artefacts. Schema migrations are gated on a
`PRAGMA user_version`; calling `setup()` on every run is idempotent.

### Concurrency

SQLite uses WAL mode (set automatically by `SQLiteHelper`), so reads
don't block writes. The `BEGIN; SELECT; DELETE; COMMIT` pattern in
`delete_pending_interrupt_atomic(...)` serializes writes via SQLite's
single-writer lock. Two resumers in the same process or different
processes contending on the same execution_id will see one winner; the
loser raises `AlreadyResumed` cleanly.

Don't share one `SQLiteCheckpointer` across processes if those processes
are on different machines — SQLite is filesystem-bound. For that, use
Postgres.

## Postgres

Ships under the `[postgres]` extra:

```bash
pip install 'fastaiagent[postgres]'
```

```python
from fastaiagent import Chain
from fastaiagent.checkpointers.postgres import PostgresCheckpointer

chain = Chain(
    "my-flow",
    checkpointer=PostgresCheckpointer(
        "postgresql://user:pass@host:5432/fastaiagent",
        schema="fastaiagent",  # default; pass a custom name to share one DB
        min_pool_size=1,
        max_pool_size=10,
    ),
)
```

### Schema versioning

`PostgresCheckpointer.setup()` reads
[`migrations/postgres_v1.sql`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/fastaiagent/checkpointers/migrations/postgres_v1.sql),
applies the DDL inside `IF NOT EXISTS` guards, and writes a row to
`schema_version`. Re-runs are no-ops. Recommended deploy step:

```bash
fastaiagent setup-checkpointer --backend postgres \
    --connection-string "$DATABASE_URL"
```

### Why `JSONB` and `TIMESTAMPTZ`?

- `JSONB` is parsed once on `INSERT` (so `state_snapshot` and
  `interrupt_context` lookups are O(log n) on indexed paths) and
  deduplicated on disk. JSON would re-parse on every read.
- `TIMESTAMPTZ` carries timezone explicitly. Production deployments
  span timezones and DST transitions; storing a `TIMESTAMP` and
  praying everyone is in UTC is how 3am pages happen.

### Atomic resume claim

The pending-interrupt claim is a single statement:

```sql
DELETE FROM fastaiagent.pending_interrupts
 WHERE execution_id = %s
RETURNING execution_id, chain_name, node_id, reason, context, agent_path, created_at
```

Postgres MVCC guarantees that of N concurrent transactions on the same
row, only one's `DELETE` succeeds; the others' `RETURNING` clause
returns no rows, and the SDK raises `AlreadyResumed`. The
[`test_postgres_concurrent_resume.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/tests/integration/test_postgres_concurrent_resume.py)
gate runs 8 threads racing to claim one row and asserts exactly one
wins on every CI build.

### Connection pool

`psycopg_pool.ConnectionPool` is opened on first use, sized
`min_pool_size .. max_pool_size`. If you're behind an HTTP service,
size it to match your worker concurrency:

```python
PostgresCheckpointer(
    DSN,
    min_pool_size=2,
    max_pool_size=os.cpu_count() * 4,
)
```

For very high concurrency, prefer external pooling (PgBouncer,
RDS Proxy) and keep the SDK pool small.

## Schema sharing

The default Postgres schema name is `fastaiagent`. Override per
checkpointer to run multiple SDK installations against one DB:

```python
prod_checkpointer = PostgresCheckpointer(DSN, schema="fa_prod")
test_checkpointer = PostgresCheckpointer(DSN, schema="fa_test")
```

Each schema is independent; checkpoints / pending interrupts /
idempotency rows don't cross.

## Cleanup

`prune(older_than=…)` deletes:

- `checkpoints` rows older than the cutoff **with status in
  `('completed', 'failed')`** — interrupted rows are preserved so
  pending HITL workflows survive.
- `idempotency_cache` rows older than the cutoff.

```python
from datetime import timedelta
from fastaiagent import SQLiteCheckpointer

cp = SQLiteCheckpointer()
cp.setup()
deleted = cp.prune(older_than=timedelta(days=14))
print(f"Pruned {deleted} rows.")
```

A nightly cron job is the typical deployment.

## Custom backends

The Protocol surface is small. To add Redis, S3, MongoDB, etc.,
implement these methods on a class and pass it as `checkpointer=`:

```python
from datetime import timedelta
from typing import Any
from fastaiagent.chain.checkpoint import Checkpoint
from fastaiagent.checkpointers import PendingInterrupt


class MyBackend:
    def setup(self) -> None: ...
    def put(self, checkpoint: Checkpoint) -> None: ...
    def put_writes(self, execution_id: str, checkpoint_id: str, writes: list) -> None: ...
    def get_last(self, execution_id: str) -> Checkpoint | None: ...
    def get_by_id(self, execution_id: str, checkpoint_id: str) -> Checkpoint | None: ...
    def list(self, execution_id: str, *, limit: int = 100) -> list[Checkpoint]: ...
    def list_pending_interrupts(self, *, limit: int = 100) -> list[PendingInterrupt]: ...
    def record_interrupt(self, checkpoint: Checkpoint, pending: PendingInterrupt) -> None: ...
    def delete_pending_interrupt_atomic(self, execution_id: str) -> PendingInterrupt | None: ...
    def delete_execution(self, execution_id: str) -> None: ...
    def get_idempotent(self, execution_id: str, function_key: str) -> Any | None: ...
    def put_idempotent(self, execution_id: str, function_key: str, result: Any) -> None: ...
    def prune(self, older_than: timedelta) -> int: ...
```

The
[`Checkpointer`](api-reference.md#checkpointer-protocol) Protocol is
`@runtime_checkable`, so `isinstance(my_backend, Checkpointer)` works
in tests. The two non-obvious methods:

- **`record_interrupt(checkpoint, pending)`** must persist both rows in
  one transaction. The `/approvals` UI must never observe a
  half-suspended workflow.
- **`delete_pending_interrupt_atomic(execution_id)`** must be atomic
  with respect to concurrent callers. Postgres uses `DELETE … RETURNING`;
  SQLite uses `BEGIN; SELECT; DELETE; COMMIT`. Redis would use a Lua
  script or a `WATCH/MULTI` transaction.

Run the
[parameterized protocol suite](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/tests/integration/test_postgres_checkpointer.py)
against your backend before relying on it in production.

## Deployment shapes

### Single-container app

SQLite on a mounted volume. Zero external dependencies. The container's
process is the only writer.

```yaml
# docker-compose.yml
services:
  app:
    image: my-app:latest
    volumes:
      - ./fastaiagent-data:/app/.fastaiagent
    environment:
      FASTAIAGENT_LOCAL_DB: /app/.fastaiagent/local.db
```

### Multi-replica HTTP service

Postgres for the checkpoint store; any replica can serve any resume.

```yaml
services:
  api:
    image: my-app:latest
    deploy:
      replicas: 3
    environment:
      DATABASE_URL: postgresql://app:pass@db:5432/myapp
  db:
    image: postgres:16-alpine
```

In your app:

```python
from fastaiagent import Chain
from fastaiagent.checkpointers.postgres import PostgresCheckpointer

chain = Chain(
    "flow",
    checkpointer=PostgresCheckpointer(os.environ["DATABASE_URL"]),
)
```

The atomic-claim contract guarantees correctness even when 3 replicas
all serve the same `POST /api/executions/{id}/resume` request
simultaneously (e.g., a webhook delivered three times).

### Worker pool

A queue (Redis, SQS, RabbitMQ) drives jobs into a worker pool. Each
worker polls for pending interrupts and resumes them.

```python
import time
from datetime import timedelta
from fastaiagent.checkpointers.postgres import PostgresCheckpointer

cp = PostgresCheckpointer(os.environ["DATABASE_URL"])

while True:
    for pending in cp.list_pending_interrupts(limit=10):
        # Domain logic: should this auto-resume? After how long? With what?
        if should_auto_approve(pending):
            try:
                chain = chains_by_name[pending.chain_name]
                await chain.aresume(pending.execution_id, resume_value=Resume(approved=True))
            except AlreadyResumed:
                pass  # another worker beat us
    time.sleep(5)
```

The atomic-claim contract makes "another worker beat us" a clean
no-op, not a duplicate side effect.

## See also

- [Multi-agent durability](multi-agent.md) — how `agent_path` segments
  compose across runner types.
- [Side effects](side-effects.md) — why some functions need `@idempotent`
  on top of a durable checkpointer.
- [API reference](api-reference.md) — exact Protocol signatures and
  return types.
