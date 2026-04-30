# Durability

Build agents that survive process crashes, pause for human approval for
days, and never re-fire side effects on resume.

```python
from fastaiagent import Chain, Resume, interrupt

def approve_refund(state):
    if state["amount"] > 10_000:
        decision = interrupt(
            reason="manager_approval",
            context={"amount": state["amount"]},
        )
        return {"approved": decision.approved}
    return {"approved": True}

# First run — suspends and exits cleanly.
result = chain.execute({"amount": 50_000}, execution_id="refund-abc")
assert result.status == "paused"

# Hours, days, or a server restart later, in a different process:
result = await chain.aresume(
    "refund-abc",
    resume_value=Resume(approved=True, metadata={"approver": "alice"}),
)
assert result.status == "completed"
```

## What v1.0 ships

- **Crash-proof agents.** A real `SIGKILL` in the middle of a 5-node
  chain. The next process call resumes at the last successful checkpoint.
- **Pause for human approval. For days.** `interrupt()` suspends a
  workflow cleanly. The process can exit. Hours later, an HTTP request,
  a CLI command, or a click in the local UI resumes it.
- **SQLite locally. Postgres in production. Same SDK.** The
  `Checkpointer` Protocol lets you swap one constructor argument and get
  multi-process / distributed durability — same semantics, same atomic
  resume claim.
- **Multi-agent durability.** Hierarchical `agent_path` so a paused tool
  inside a worker inside a supervisor renders as
  `supervisor:planner/worker:auditor/tool:approve_refund` everywhere —
  in checkpoints, in the `/approvals` UI, and in resume errors.
- **Built-in side-effect protection.** The `@idempotent` decorator
  caches a function's result for the lifetime of an execution — wrap
  your `charge_customer` once, never double-charge again.

## Pick a path

| Goal | Read |
|---|---|
| Get a paused-and-resumed chain running in 5 minutes | [Quickstart](quickstart.md) |
| Understand why side effects double-fire on resume — and how to fix it | [Side effects & idempotency](side-effects.md) |
| Apply durability to common production shapes | [Patterns](patterns.md) |
| Wire `interrupt()` into Agent / Swarm / Supervisor | [Multi-agent durability](multi-agent.md) |
| Choose between SQLite and Postgres for production | [Checkpointers](checkpointers.md) |
| Look up exact signatures, types, return shapes | [API reference](api-reference.md) |
| Coming from LangGraph? | [Migrating from LangGraph](../migration-guides/from-langgraph.md) |

## How it works in one paragraph

When a node calls `interrupt(reason, context)`, the chain executor catches
the signal, persists an `interrupted` checkpoint and a row in
`pending_interrupts` in **one transaction**, and returns
`ChainResult(status="paused")`. The Python process can exit. To resume,
any process with access to the same checkpoint store calls
`chain.aresume(execution_id, resume_value=Resume(...))`. The resumer
**atomically deletes** the pending row (Postgres `DELETE … RETURNING *`,
SQLite `BEGIN; SELECT; DELETE; COMMIT`), re-enters the suspended node
with the resume value injected via a `ContextVar`, and `interrupt()`
returns the value instead of raising. Concurrent resumers — a
double-clicked Approve button, two webhook deliveries — see
`AlreadyResumed`. Same machinery powers Agent / Swarm / Supervisor; the
only difference is the segments their `agent_path` carries.
