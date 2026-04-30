# Durability API reference

Every primitive the v1.0 durability surface ships, with exact signatures
and the contract each one upholds. For tutorials, start with
[Quickstart](quickstart.md). For end-to-end shapes, see
[Patterns](patterns.md).

## Public exports

All of these are importable directly from the top-level `fastaiagent`
package:

```python
from fastaiagent import (
    # Checkpointers
    Checkpointer,            # Protocol — implement to add a backend.
    SQLiteCheckpointer,      # Default backend.
    PendingInterrupt,        # Pydantic model for /api/pending-interrupts rows.

    # Suspension primitives
    interrupt,               # Call from any node to suspend.
    Resume,                  # Value passed to aresume().
    InterruptSignal,         # Internal control-flow exception.
    AlreadyResumed,          # Raised on a stale claim.

    # Side-effect protection
    idempotent,              # Decorator.
    IdempotencyError,        # Raised on non-JSON-serializable returns.
)
```

`PostgresCheckpointer` is **lazily** importable and only loads psycopg
when actually used:

```python
from fastaiagent.checkpointers.postgres import PostgresCheckpointer
# requires `pip install 'fastaiagent[postgres]'`
```

## `interrupt`

```python
def interrupt(reason: str, context: dict[str, Any]) -> Resume: ...
```

Suspend the current workflow. First call (no resume value in scope):
raises `InterruptSignal`. The chain / agent executor catches it,
persists an `interrupted` checkpoint plus a row in `pending_interrupts`,
and returns `ChainResult(status="paused")` /
`AgentResult(status="paused")`. Second call (after `chain.aresume(...)`):
returns the `Resume` value the resumer passed in.

The `context` dict is **JSON-serialized at suspend time** and frozen
into the checkpoint and the pending row. The resumer always sees the
original snapshot — context is never recomputed. See
[the frozen-context invariant](side-effects.md#the-frozen-context-invariant).

## `Resume`

```python
class Resume(BaseModel):
    approved: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
```

Value passed to `chain.aresume(...)` /
`agent.aresume(...)` / `swarm.aresume(...)` /
`supervisor.aresume(...)` and returned by `interrupt()` on resume.

The `metadata` dict is the structured channel for non-decision data:
`{"approver": "alice", "reason": "verified ID", "ticket": "T-1234"}`.
There's also a reserved `data` field for non-approval resume cases that
v1.0 does not exercise.

## `InterruptSignal`

```python
class InterruptSignal(Exception):
    reason: str
    context: dict[str, Any]
```

Raised by `interrupt()` when no resume value is in scope. The chain /
agent / swarm / supervisor executors catch it; user code should not.

## `AlreadyResumed`

```python
class AlreadyResumed(Exception): ...
```

Raised by `chain.aresume(...)` (and friends) when the
`pending_interrupts` row was already claimed by another resumer.
Concurrency safety net: a double-clicked Approve button, two webhook
deliveries of the same payload, or two replicas racing on the same
execution all converge here.

The HTTP `POST /api/executions/{id}/resume` endpoint maps this to
`409 Conflict`. The CLI's `fastaiagent resume` exits with code 2.

## `Chain.aresume`

```python
async def aresume(
    self,
    execution_id: str,
    *,
    resume_value: Resume | None = None,
    modified_state: dict[str, Any] | None = None,
) -> ChainResult: ...
```

Async alias for `Chain.resume(...)` — exists so `Chain` matches the
`Agent` / `Swarm` / `Supervisor` surface, letting the HTTP / CLI
entrypoints treat all four runner types uniformly.

For an interrupted checkpoint: pass `resume_value`. The
`pending_interrupts` row is atomically claimed before the suspended
node re-executes. Concurrent resumers see `AlreadyResumed`.

For a failed checkpoint (no `interrupt()`, just a regular exception):
pass `modified_state` to patch chain state before the next node runs.
This is the v0.x behavior, preserved.

## `Agent.aresume`

```python
async def aresume(
    self,
    execution_id: str,
    *,
    resume_value: Resume | None = None,
    context: RunContext[Any] | None = None,
    agent_path_prefix: str | None = None,
    **kwargs: Any,
) -> AgentResult: ...
```

Three resume shapes are auto-detected from the latest checkpoint:

1. **Interrupted** — pass `resume_value`. Claims the pending row,
   re-invokes the suspended tool with `_resume_value` in scope.
2. **Tool-boundary crash** — saved tool re-invoked with saved args; LLM
   is **not** re-called.
3. **Turn-boundary crash** — loop re-enters at the saved turn,
   re-issuing the LLM call.

`agent_path_prefix` is an advanced kwarg used by `Supervisor`'s
delegate tools to scope the resume to a worker's subtree (so the
worker doesn't accidentally pick up a sibling supervisor pre-tool
checkpoint as "latest").

## `Swarm.aresume`

```python
async def aresume(
    self,
    execution_id: str,
    *,
    resume_value: Resume | None = None,
    context: RunContext[Any] | None = None,
    **kwargs: Any,
) -> AgentResult: ...
```

Determines the active agent from the latest checkpoint's `agent_path`,
recovers `SwarmState` from the most recent `handoff:N` boundary, then
either:

- calls `agent.aresume(...)` (interrupt resume), or
- calls `agent.arun(...)` (crash recovery — re-issues LLM).

After the active agent returns, the swarm loop continues with
remaining handoffs, allowlists, and `max_handoffs` enforced.

## `Supervisor.aresume`

```python
async def aresume(
    self,
    execution_id: str,
    *,
    resume_value: Resume | None = None,
    context: RunContext[Any] | None = None,
    **kwargs: Any,
) -> AgentResult: ...
```

Recovers the original supervisor input from the supervisor's earliest
checkpoint, binds `_resume_value` in a ContextVar, and re-runs
`supervisor.arun(...)`. The supervisor's LLM is re-issued; each
`delegate_to_<role>` tool detects existing worker state and resumes
that worker instead of running it fresh.

## `idempotent`

```python
def idempotent(
    fn: Callable[..., Any] | None = None,
    *,
    key_fn: Callable[..., str] | None = None,
) -> Callable[..., Any]: ...
```

Decorator. Cache the wrapped function's result by
`(execution_id, sha256(qualname + args + kwargs))` so a re-executed
node — typically the suspended node on resume from `interrupt()` —
does not re-run the side effect.

```python
@idempotent
def charge_customer(amount, customer_id):
    return stripe.charge(...)

@idempotent(key_fn=lambda user, req: f"{user.id}:{req.id}")
def process(user, req):
    ...
```

- First call inside an execution: runs the body, stores the result.
- Subsequent calls in the same execution with the same key: returns
  the cached value, never re-runs the body.
- Calls in a different execution_id: cache miss, runs again.
- Calls outside any chain run: no caching, body runs every time.

Returns are stored via `pydantic_core.to_jsonable_python`, so Pydantic
models and dataclasses round-trip cleanly. Non-JSONable returns raise
`IdempotencyError` at the first call.

A consequence: the first call returns the original Python object; a
**cache hit** returns the JSON-deserialized form (a dict, list, or
primitive). Design idempotent functions to return plain data, or
hydrate the cached dict back into a model at the call site.

## `IdempotencyError`

```python
class IdempotencyError(Exception): ...
```

Raised when an `@idempotent` function returns a value that is not
JSON-serializable (e.g. an open file handle, a live socket). Wrap the
return in a Pydantic model, return plain data, or split this into a
non-cached node.

## `Checkpointer` (Protocol)

```python
@runtime_checkable
class Checkpointer(Protocol):
    def setup(self) -> None: ...
    def put(self, checkpoint: Checkpoint) -> None: ...
    def put_writes(self, execution_id: str, checkpoint_id: str, writes: list[Any]) -> None: ...
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

Two contract requirements that are not in the type signatures:

- **`record_interrupt(...)` must persist both rows in one transaction.**
  The `/approvals` UI must never observe a half-suspended workflow
  (interrupted checkpoint without a `pending_interrupts` row, or vice
  versa).
- **`delete_pending_interrupt_atomic(...)` must be atomic with respect
  to concurrent callers.** Of N callers on the same execution_id,
  exactly one returns the row; the rest return `None`. Postgres uses
  `DELETE … RETURNING *`; SQLite uses `BEGIN; SELECT; DELETE; COMMIT`.

`prune(older_than)` must skip `interrupted` checkpoints — pending HITL
workflows that have been waiting longer than the cutoff are preserved.

## `SQLiteCheckpointer`

```python
class SQLiteCheckpointer:
    def __init__(self, db_path: str | None = None) -> None: ...
```

Default backend. If `db_path` is omitted, uses
`get_config().resolved_checkpoint_db_path` (the configured local DB,
defaulting to `./.fastaiagent/local.db`). Schema lives in the unified
local.db; calling `setup()` is idempotent (gated on
`PRAGMA user_version`).

Adds two non-protocol helpers used internally by `@idempotent`:
`get_idempotent`, `put_idempotent`. The Postgres backend exposes the
same shape.

## `PostgresCheckpointer`

```python
class PostgresCheckpointer:
    def __init__(
        self,
        connection_string: str,
        *,
        schema: str = "fastaiagent",
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None: ...
```

Ships under the `[postgres]` extra. Uses psycopg3 + `psycopg_pool`.
Stores all JSON columns as `JSONB` and timestamps as `TIMESTAMPTZ`.
The pending-interrupt claim is a single
`DELETE … RETURNING *` statement.

`schema` must match `[A-Za-z_][A-Za-z0-9_]*` (the SDK validates this
to keep the migration template safe).

## `PendingInterrupt`

```python
class PendingInterrupt(BaseModel):
    execution_id: str
    chain_name: str
    node_id: str
    reason: str
    context: dict[str, Any]
    agent_path: str | None
    created_at: str
```

The shape returned by `Checkpointer.list_pending_interrupts()` and
`delete_pending_interrupt_atomic()`. Identical to the row the
`/approvals` UI page renders.

## `ChainResult` / `AgentResult`

Both gained two fields in v1.0:

```python
class ChainResult(BaseModel):
    # ... v0.x fields ...
    status: str = "completed"             # "completed" or "paused"
    pending_interrupt: dict[str, Any] | None = None

class AgentResult(BaseModel):
    # ... v0.x fields ...
    execution_id: str = ""
    status: str = "completed"             # "completed" or "paused"
    pending_interrupt: dict[str, Any] | None = None
```

`pending_interrupt` (when `status == "paused"`) carries the same
payload as the corresponding `pending_interrupts` row:
`{reason, context, node_id, agent_path}`. The `/approvals` UI reads
both and renders the same data.

## HTTP endpoints (Local UI)

Mounted by [`build_app(...)`](../ui/index.md). Pass an iterable of
`runners=` so the server can resume them.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/executions/{execution_id}` | Full checkpoint history. Returns `404` on unknown id. |
| `GET` | `/api/pending-interrupts` | List of pending rows (paginated by `?limit=`). |
| `POST` | `/api/executions/{execution_id}/resume` | Atomic resume. Body: `{"approved": bool, "metadata": {...}, "reason": "?"}`. Returns `409` on `AlreadyResumed`, `503` if no runner is registered for this `chain_name`. |
| `GET` | `/api/overview` | Home dashboard counts; includes `pending_approvals_count` and `failed_executions_count`. |

## CLI

| Command | Purpose |
|---|---|
| `fastaiagent resume <id> --runner module:attr [--value JSON]` | Resume via Python entrypoint. Exit code `2` on `AlreadyResumed`. |
| `fastaiagent list-pending [--db-path PATH]` | Rich table of pending interrupts. |
| `fastaiagent inspect <id> [--db-path PATH]` | Checkpoint history for one execution. Exit code `1` if the execution is unknown. |
| `fastaiagent setup-checkpointer --backend [sqlite\|postgres] --connection-string ...` | Provision / verify the backend's schema. Idempotent. |

## ContextVars (advanced)

For users implementing custom topologies or backends. These live in
`fastaiagent.chain.interrupt`:

| ContextVar | Set by | Read by |
|---|---|---|
| `_execution_id: ContextVar[str \| None]` | `execute_chain`, `Agent._arun_core`, `Swarm._arun_swarm`, `Supervisor.arun` | `interrupt()`, `@idempotent`, `record_interrupt(...)` |
| `_resume_value: ContextVar[Resume \| None]` | `Chain.aresume`, `Agent.aresume`, `Swarm.aresume`, `Supervisor.aresume` (just before re-invoking the suspended node) | `interrupt()` (returns this when set) |
| `_agent_path: ContextVar[str \| None]` | Each runner extends with its segment | `record_interrupt(...)`, `_put_*_checkpoint` helpers |
| `_current_checkpointer: ContextVar[Checkpointer \| None]` | `execute_chain` (Phase 3 wiring) | `@idempotent`'s `inner(...)` |

In normal usage you never read or set these directly — the runners do
it for you.
