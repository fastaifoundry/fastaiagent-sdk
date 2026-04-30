# Migrating from LangGraph

This guide maps LangGraph concepts to FastAIAgent equivalents and shows
how to migrate. The two SDKs converge on the same idea — graphs of nodes
with a durable execution model — but differ in shape, ergonomics, and
operational surface.

## Feature mapping

| LangGraph | FastAIAgent | Notes |
|---|---|---|
| `StateGraph` | [`Chain`](../chains/index.md) | Both are directed graphs over typed state. FastAIAgent's `Chain` adds cyclic edges with explicit `max_iterations` + `exit_condition`. |
| `MessagesState` | [`ChainState`](../chains/checkpointing.md) | FastAIAgent state is a free-form dict by default; pass `state_schema=` for JSON-schema validation. |
| `START` / `END` constants | First / last node in topological order | Implicit, not literal. |
| `add_node(name, fn)` | `chain.add_node(name, agent=…)` / `chain.add_node(name, tool=…, type=NodeType.tool)` | FastAIAgent distinguishes Agent / Tool / HITL / Transformer / Condition / Parallel node types. |
| `add_edge(src, dst)` | `chain.connect(src, dst)` | Plus `condition=`, `max_iterations=`, `exit_condition=`. |
| `compile(checkpointer=…)` | Pass `checkpointer=…` to the `Chain` constructor | One step, not two. |
| `MemorySaver` | (use `SQLiteCheckpointer(":memory:")` for tests) | Memory is fine for tests; production runs through a real DB. |
| `SqliteSaver` | [`SQLiteCheckpointer`](../chains/checkpointing.md#custom-checkpoint-backend) | Same backend, same idea. SDK ships it as the default. |
| `PostgresSaver` | [`PostgresCheckpointer`](../durability/checkpointers.md#postgres) | Native psycopg3, JSONB, atomic claim via `DELETE … RETURNING`. |
| `interrupt()` | [`interrupt()`](../durability/api-reference.md#interrupt) | Same name, same idea. FastAIAgent's `Resume(approved, metadata)` is structured; LangGraph passes a free-form dict. |
| `Command(resume=…)` | `chain.aresume(execution_id, resume_value=Resume(…))` | FastAIAgent's resume is async-by-default; sync wrapper available. |
| `graph.invoke(...)` | `chain.execute(...)` (sync) or `await chain.aexecute(...)` (async) | |
| `graph.astream(...)` | `await chain.aexecute(..., trace=True)` for OTel spans; future v1.x for token-level streaming on chains | Agent.astream / Swarm.astream exist today for token streaming. |
| `Send(...)` | Cyclic edges + condition routing on the chain | FastAIAgent doesn't have a native fan-out primitive at the chain level (parallel exists at the node level). |
| `interrupt_before` / `interrupt_after` on edges | `interrupt()` inside the relevant node | FastAIAgent prefers explicit `interrupt()` calls over edge-level interception — the call site is the audit-log entry. |

## A complete migration

### LangGraph

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command


class State(TypedDict):
    amount: int
    approved: bool


def review(state):
    decision = interrupt({"amount": state["amount"]})
    return {"approved": decision["approved"]}


def issue(state):
    if state["approved"]:
        # ...issue refund
        pass
    return state


graph = (
    StateGraph(State)
    .add_node("review", review)
    .add_node("issue", issue)
    .add_edge(START, "review")
    .add_edge("review", "issue")
    .add_edge("issue", END)
    .compile(checkpointer=SqliteSaver.from_conn_string("./graph.db"))
)

config = {"configurable": {"thread_id": "refund-abc"}}
graph.invoke({"amount": 50_000, "approved": False}, config=config)

# ... later ...
graph.invoke(Command(resume={"approved": True}), config=config)
```

### FastAIAgent

```python
from fastaiagent import Chain, FunctionTool, Resume, SQLiteCheckpointer, interrupt
from fastaiagent.chain.node import NodeType


def review(amount: str) -> dict:
    decision = interrupt(
        reason="manager_approval",
        context={"amount": int(amount)},
    )
    return {"approved": decision.approved}


def issue(approved: str) -> dict:
    return {"refund_issued": str(approved).lower() == "true"}


chain = Chain(
    "refund-flow",
    checkpointer=SQLiteCheckpointer(db_path="./fastaiagent.db"),
)
chain.add_node(
    "review",
    tool=FunctionTool(name="review_tool", fn=review),
    type=NodeType.tool,
    input_mapping={"amount": "{{state.amount}}"},
)
chain.add_node(
    "issue",
    tool=FunctionTool(name="issue_tool", fn=issue),
    type=NodeType.tool,
    input_mapping={"approved": "{{state.output.approved}}"},
)
chain.connect("review", "issue")

paused = chain.execute({"amount": 50_000}, execution_id="refund-abc")
assert paused.status == "paused"

# ... later ...
result = await chain.aresume("refund-abc", resume_value=Resume(approved=True))
assert result.status == "completed"
```

## Key differences

### 1. `execution_id`, not `thread_id`

FastAIAgent calls the run identifier `execution_id`. LangGraph calls
the same thing `thread_id`. We chose `execution_id` because
"thread" overloads in Python and in chat product taxonomies.

### 2. `Resume` is structured

LangGraph's `Command(resume=value)` accepts any value. FastAIAgent's
`Resume(approved=bool, metadata=dict)` is a Pydantic model — callable
shapes are a structured Approve/Reject decision plus metadata.

The structure pays off in two places:

- The `/approvals` UI renders Approve / Reject buttons that map to
  `approved=True` / `approved=False` — no schema discovery needed.
- Audit logs have a consistent shape across all suspended workflows.

If you need free-form data, use `metadata`:

```python
Resume(approved=True, metadata={"user_decision": ..., "context": ...})
```

### 3. Atomic resume claim is built in

LangGraph's checkpointers are storage; the resume coordination logic is
in user code. FastAIAgent's
[`Checkpointer.delete_pending_interrupt_atomic(...)`](../durability/api-reference.md#checkpointer-protocol)
makes "exactly one resumer wins" a backend-level guarantee. A
double-clicked Approve button raises `AlreadyResumed` deterministically;
the HTTP layer maps it to `409 Conflict`; the CLI exits with code 2.

### 4. `@idempotent` ships with the SDK

LangGraph leaves side-effect protection to the user. FastAIAgent ships
[`@idempotent`](../chains/idempotency.md) — wrap a function once,
its result is cached for the lifetime of an execution. Resume re-runs
the node, but the wrapped function returns its cached result instead
of re-firing.

```python
@idempotent
def charge_card(amount):
    return stripe.charge(amount=amount)
```

### 5. The `/approvals` UI is included

FastAIAgent ships a local web UI ([`fastaiagent ui start`](../ui/index.md))
with built-in pages for:

- `/approvals` — list every paused workflow, click to inspect the
  frozen context, click Approve / Reject.
- `/executions/:id` — full checkpoint history for any run.
- Two new home KPI cards: Pending approvals + Failed executions.

LangSmith, LangServe, or a custom UI is what LangGraph users typically
build for this.

### 6. Hierarchical `agent_path`

For multi-agent topologies, FastAIAgent attaches a hierarchical
`agent_path` to every checkpoint:

| Topology | Path |
|---|---|
| Agent | `agent:<name>/tool:<t>` |
| Swarm | `swarm:<s>/agent:<a>/tool:<t>` |
| Supervisor | `supervisor:<s>/worker:<r>/tool:<t>` |

So a paused tool inside a worker inside a supervisor renders in the
`/approvals` UI as `supervisor:planner/worker:auditor/tool:approve_refund`
— the operator sees the full delegation path immediately.

LangGraph subgraphs nest similarly but don't standardize on a string
representation; the UI / observability layer is left to the user.

### 7. Postgres support is a one-line swap

Switching from SQLite to Postgres in LangGraph means changing the
`checkpointer` argument **and** confirming your custom code paths
match. In FastAIAgent the
[`Checkpointer` Protocol](../durability/api-reference.md#checkpointer-protocol)
is identical between backends; the parameterized integration suite
([`tests/integration/test_postgres_checkpointer.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/tests/integration/test_postgres_checkpointer.py))
runs the same 13 protocol tests against both backends on every CI
build, so drift surfaces immediately.

```python
# Local development
chain = Chain("flow", checkpointer=SQLiteCheckpointer())

# Production
chain = Chain("flow", checkpointer=PostgresCheckpointer(DSN))
```

## Migration checklist

1. Rename `thread_id` → `execution_id` everywhere.
2. Replace `MemorySaver` / `SqliteSaver` with `SQLiteCheckpointer()`.
3. Replace `PostgresSaver` with `PostgresCheckpointer(...)` — `psycopg3`
   is required (no psycopg2 path in v1.0).
4. Replace `interrupt(value)` with `interrupt(reason=..., context=...)`.
5. Replace `Command(resume=...)` with
   `chain.aresume(execution_id, resume_value=Resume(...))`.
6. Wrap any side-effectful node functions with
   [`@idempotent`](../chains/idempotency.md) — or pass provider-side
   idempotency keys derived from `execution_id`.
7. If you used `interrupt_before` / `interrupt_after` on edges, move
   the `interrupt()` call **into the node body** at the right point.
   The call site becomes the audit-log entry.

## What FastAIAgent does NOT have (yet)

- **Watchdog / auto-resumption.** LangGraph users sometimes pair their
  graphs with a scheduler; FastAIAgent leaves that to your existing
  cron / queue infrastructure. The `fastaiagent list-pending` CLI is
  the supported integration point.
- **Subgraphs as first-class objects.** FastAIAgent's `Chain` doesn't
  embed other Chains as nodes. Use `Swarm` or `Supervisor` for
  multi-agent topologies.
- **Async checkpoint writes.** Checkpoint writes are sync (psycopg3
  sync, `sqlite3`). The runner methods are `async def` for
  consistency, but the actual store calls block. Async psycopg lands
  in v1.1.

## See also

- [Durability quickstart](../durability/quickstart.md) — paste-and-run
  example.
- [Multi-agent durability](../durability/multi-agent.md) — `agent_path`
  composition across runner types.
- [Side effects & idempotency](../durability/side-effects.md) — the
  footgun every team hits when migrating from a non-durable framework.
