# Quickstart

A complete pause-and-resume in two short scripts. No real LLM, no real
HTTP — just `Chain` + `interrupt()` + `chain.aresume(...)`.

## 1. Install

```bash
pip install fastaiagent
```

## 2. Build a chain that suspends

`refund.py`:

```python
from typing import Any
from fastaiagent import Chain, FunctionTool, SQLiteCheckpointer, interrupt
from fastaiagent.chain.node import NodeType


def approve_refund(amount: str) -> dict[str, Any]:
    """High-value refunds suspend for human approval."""
    n = int(amount)
    if n > 10_000:
        decision = interrupt(
            reason="manager_approval",
            context={"amount": n},
        )
        return {
            "approved": decision.approved,
            "approver": decision.metadata.get("approver"),
        }
    return {"approved": True, "auto": True}


def issue_refund(approved: str) -> dict[str, Any]:
    """No-op terminal node; in production this would call a payments API."""
    return {"refund_issued": str(approved).lower() == "true"}


def build_chain() -> Chain:
    chain = Chain(
        "refund-flow",
        checkpointer=SQLiteCheckpointer(),  # writes to ./.fastaiagent/local.db
    )
    chain.add_node(
        "approve",
        tool=FunctionTool(name="approve_tool", fn=approve_refund),
        type=NodeType.tool,
        input_mapping={"amount": "{{state.amount}}"},
    )
    chain.add_node(
        "issue",
        tool=FunctionTool(name="issue_tool", fn=issue_refund),
        type=NodeType.tool,
        input_mapping={"approved": "{{state.output.approved}}"},
    )
    chain.connect("approve", "issue")
    return chain
```

## 3. Run — and watch it pause

```python
# pause.py
from refund import build_chain

chain = build_chain()
result = chain.execute({"amount": 50_000}, execution_id="refund-abc")

print(result.status)            # "paused"
print(result.pending_interrupt) # {"reason": "manager_approval", "context": {"amount": 50000}, ...}
```

The Python process can exit now. The pause checkpoint and the
`pending_interrupts` row are committed.

## 4. Resume — even from a different process

```python
# resume.py
from fastaiagent import Resume
from fastaiagent._internal.async_utils import run_sync
from refund import build_chain

chain = build_chain()
result = run_sync(chain.aresume(
    "refund-abc",
    resume_value=Resume(approved=True, metadata={"approver": "alice"}),
))

print(result.status)                              # "completed"
print(result.final_state["output"])               # {"refund_issued": True}
```

This works whether `pause.py` is still running, exited cleanly, or was
SIGKILLed mid-flight. The `Chain` rebuilt in `resume.py` reads the
checkpoint store, claims the `pending_interrupts` row atomically, and
re-enters the `approve` node with the `Resume` value in scope.

A second `aresume(...)` call on the same execution_id raises
[`AlreadyResumed`](api-reference.md#alreadyresumed) — the safety net
against double-clicked Approve buttons.

## 5. Inspect from the CLI

```bash
fastaiagent list-pending          # see every paused workflow
fastaiagent inspect refund-abc    # checkpoint history + statuses
```

## What you've built

- A workflow that **suspends without crashing** when a node calls
  `interrupt()`.
- A resume path that survives process restarts because the messages and
  pending interrupt are persisted in a real database.
- An **atomic resume claim** that makes a double-resume into a clean
  `AlreadyResumed` error, not a duplicate side effect.

## Where to go next

- [Side effects & idempotency](side-effects.md) — the footgun that bites
  every team adding durability for the first time.
- [Patterns](patterns.md) — how to apply this shape to refunds, payments,
  multi-step orchestration, etc.
- [Multi-agent durability](multi-agent.md) — the same primitives in
  `Agent` / `Swarm` / `Supervisor`.
- [Checkpointers](checkpointers.md) — swapping in Postgres for
  multi-process production.
