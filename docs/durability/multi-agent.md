# Multi-agent durability

The same `interrupt()` / `aresume()` machinery powers all four runner
shapes: `Chain`, `Agent`, `Swarm`, `Supervisor`. The only difference is
the segments their `agent_path` carries — and that controls how
`/approvals` displays them and how resumes find the right node.

## The agent_path hierarchy

| Topology | Path shape |
|---|---|
| Chain | (no agent_path; `node_id` is the rendezvous point) |
| Agent | `agent:<name>` → `agent:<name>/tool:<tool>` |
| Swarm | `swarm:<s>` → `swarm:<s>/agent:<a>` → `swarm:<s>/agent:<a>/tool:<t>` |
| Supervisor | `supervisor:<s>` → `supervisor:<s>/worker:<r>` → `supervisor:<s>/worker:<r>/tool:<t>` |

Each level is set automatically. `Agent` extends parent path; `Swarm`
prefixes its inner agent runs; `Supervisor` overrides the inner-Agent's
segment to `supervisor:<name>` and clones each worker with
`agent_path_label="worker:<role>"`. You don't compose these yourself.

## Agent

A single `Agent` with tools. Crash-recovers at turn boundaries and
suspends cleanly when a tool calls `interrupt()`.

```python
from fastaiagent import Agent, FunctionTool, SQLiteCheckpointer, Resume, interrupt


def approve_refund(amount: int):
    if amount > 10_000:
        d = interrupt(reason="manager_approval", context={"amount": amount})
        return {"approved": d.approved}
    return {"approved": True}


agent = Agent(
    name="refund-bot",
    system_prompt="...",
    llm=...,
    tools=[FunctionTool(name="approve_refund", fn=approve_refund)],
    checkpointer=SQLiteCheckpointer(),
)

result = agent.run("Process a $50k refund.", execution_id="job-1")
# result.status == "paused"
# result.pending_interrupt["agent_path"] == "agent:refund-bot/tool:approve_refund"

# Hours later, in any process:
result = await agent.aresume("job-1", resume_value=Resume(approved=True))
```

**Three resume shapes** (covered by Phase 5's
`test_gate_agent_durability`):

1. **Interrupted run** (`status="interrupted"` checkpoint, pending row):
   pass `resume_value`. The pending row is atomically claimed; the
   suspended tool re-executes with `_resume_value` in scope so
   `interrupt()` returns the value.
2. **Tool-boundary crash** (latest checkpoint is `turn:N/tool:X`,
   `status="completed"`): the saved tool is re-invoked with the saved
   args; the LLM is **not** re-called — the assistant's tool_calls are
   already in messages.
3. **Turn-boundary crash** (latest checkpoint is `turn:N`): the loop
   re-enters at iteration N, re-issuing the LLM call.

> **Cost note:** turn-boundary resume re-issues the LLM call. Wrap
> side-effectful tool functions with [`@idempotent`](../chains/idempotency.md)
> to absorb the re-execution.

## Swarm

A peer-to-peer multi-agent topology. Every handoff between agents writes
a swarm-level checkpoint with the active agent + shared blackboard.

```python
from fastaiagent import Agent, Swarm, SQLiteCheckpointer

researcher = Agent(name="researcher", ...)
analyst    = Agent(name="analyst",    ...)
reporter   = Agent(name="reporter",   ...)

swarm = Swarm(
    name="content_team",
    agents=[researcher, analyst, reporter],
    entrypoint="researcher",
    handoffs={"researcher": ["analyst"], "analyst": ["reporter"], "reporter": []},
    checkpointer=SQLiteCheckpointer(),
)

result = swarm.run("Investigate the topic.", execution_id="trip-456")
# If a tool inside any agent calls interrupt(), result.status == "paused"
# and result.pending_interrupt.agent_path is e.g.
# "swarm:content_team/agent:analyst/tool:approve_publish".

result = await swarm.aresume("trip-456", resume_value=Resume(approved=True))
```

Resume from a swarm:

- **Interrupted**: pending row claimed atomically; the active agent's
  suspended tool runs once with the resume value injected; the loop
  continues through any remaining handoffs.
- **Crash mid-handoff**: the swarm parses the active agent from the
  latest checkpoint's `agent_path`, recovers `SwarmState` from the most
  recent `handoff:N` checkpoint, and re-runs the active agent (LLM
  re-issued for crash recovery; tools wrapped with `@idempotent` are
  not re-fired).

## Supervisor / Worker

Hierarchical delegation. The supervisor's LLM picks a worker; the worker
runs a turn loop; if a worker's tool calls `interrupt()`, the suspension
propagates up with the full nested `agent_path`.

```python
from fastaiagent import Agent, SQLiteCheckpointer
from fastaiagent.agent.team import Supervisor, Worker

planner = Supervisor(
    name="planner",
    llm=...,
    workers=[
        Worker(agent=Agent(name="researcher", ...), role="researcher", ...),
        Worker(agent=Agent(name="auditor",    ...), role="auditor",    ...),
    ],
    checkpointer=SQLiteCheckpointer(),
)

result = planner.run("Investigate and report.", execution_id="job-42")
# If auditor.tools[0] calls interrupt(), result.pending_interrupt.agent_path is
# "supervisor:planner/worker:auditor/tool:approve_refund".

result = await planner.aresume("job-42", resume_value=Resume(approved=True))
```

The supervisor's resume **re-issues its LLM** (the supervisor's LLM is
the orchestrator that decided to delegate to auditor in the first
place). Each `delegate_to_<role>` tool is durability-aware: it detects
existing worker state for the active execution and routes to the
worker's `aresume(...)` (with the supervisor's `_resume_value` in
scope) instead of running the worker fresh.

> **Cost trade-off:** The supervisor's LLM is re-issued on every
> resume, just like Chain and Swarm. With deterministic-mock LLMs in
> tests this is free; in production this means an extra inference call
> per resume. If your supervisor is doing expensive reasoning, lower
> its `temperature` to keep its decisions reproducible.

## When `interrupt()` propagates between layers

| Scenario | What suspends | `agent_path` on the pending row |
|---|---|---|
| A Chain node calls `interrupt()` | The Chain | `null` |
| A standalone Agent's tool calls `interrupt()` | The Agent | `agent:<name>/tool:<t>` |
| A tool inside a Swarm member calls `interrupt()` | The Swarm | `swarm:<s>/agent:<a>/tool:<t>` |
| A tool inside a Supervisor's worker calls `interrupt()` | The Supervisor | `supervisor:<s>/worker:<r>/tool:<t>` |

In every case, **only one** `pending_interrupts` row is written (the
innermost catcher persists; outer layers re-raise `_AgentInterrupted` so
the pending row is never duplicated). The `/approvals` UI displays the
full path so an operator immediately sees which workflow needs them.

## What is not yet supported

A few intentional non-features in v1.0; see the v1 spec's "What NOT to
build" section for the reasoning:

- **Watchdog / auto-resumption** of paused workflows. v1.0 is human-
  triggered. A scheduler cron + `fastaiagent list-pending` is the
  recommended shape if you need automated retries.
- **Distributed locks beyond `DELETE … RETURNING`.** Two resumers in two
  Postgres-backed processes race; one wins, one sees `AlreadyResumed`.
  That's enough for the human-approval case. SELECT-FOR-UPDATE-SKIP-LOCKED
  for batch resumes is on the v1.1 list.
- **Async durability.** All checkpoint writes are synchronous (psycopg3
  sync, `sqlite3`). The `aresume()` methods are `async def` for
  consistency with the rest of the SDK, but the actual store calls
  block. Async psycopg lands in v1.1.
- **Redis / S3 backends.** SQLite covers local; Postgres covers
  production. If you have a real Redis-or-S3 use case, the
  [`Checkpointer` Protocol](api-reference.md#checkpointer-protocol) is small —
  ~10 methods. Implement it.

## See also

- [Chains – Human-in-the-Loop](../chains/hitl.md) — `interrupt()` at the
  Chain level.
- [Agents – Durability](../agents/durability.md) — Agent-specific
  resume shapes.
- [Side effects](side-effects.md) — why every multi-agent resume needs
  `@idempotent` somewhere.
