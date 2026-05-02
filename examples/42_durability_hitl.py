"""Example 42 — Durable human-in-the-loop with interrupt/resume.

Demonstrates the v1.0 durability primitives:

* :func:`fastaiagent.interrupt` suspends a workflow waiting on a human
  decision. The pause is committed to a real database — the process can
  exit, crash, or be redeployed and the suspension survives.
* :meth:`Chain.aresume` re-enters the chain with a structured
  :class:`Resume` value. The atomic-claim contract makes a double
  resume raise :class:`AlreadyResumed` instead of double-firing.
* :func:`fastaiagent.idempotent` absorbs the side-effect re-execution
  inherent to replay semantics. Wrap any non-idempotent call.

After running, open ``fastaiagent ui`` and visit
``/executions/<execution_id>`` to see the checkpoint inspector — a
vertical timeline with state diff and idempotency cache. The expected
output is captured in
``docs/ui/screenshots/sprint1-3-checkpoint-timeline.png``; see
``docs/ui/checkpoint-inspector.md`` for a walkthrough.

Prereqs:
    pip install fastaiagent
"""

from __future__ import annotations

from typing import Any

from fastaiagent import (
    AlreadyResumed,
    Chain,
    FunctionTool,
    Resume,
    SQLiteCheckpointer,
    idempotent,
    interrupt,
)
from fastaiagent._internal.async_utils import run_sync
from fastaiagent.chain.node import NodeType

_charge_count: dict[str, int] = {"calls": 0}


@idempotent
def charge_card(amount: int, account: str) -> dict[str, Any]:
    """Stand-in for a payment gateway call. Wrapped so the resume replay
    does not double-charge — every call after the first one in this
    execution returns the cached receipt."""
    _charge_count["calls"] += 1
    return {"id": f"ch_{account}_{amount}", "amount": amount}


def review_step(amount: str, account: str) -> dict[str, Any]:
    """High-value charges suspend for manager approval."""
    n = int(amount)
    if n > 10_000:
        decision = interrupt(
            reason="manager_approval",
            context={"amount": n, "account": account},
        )
        if not decision.approved:
            return {"approved": False, "reason": "manager rejected"}
    receipt = charge_card(n, account)
    return {
        "approved": True,
        "charge_id": receipt["id"],
        "approver": _approver_name(),
    }


def _approver_name() -> str:
    """Read the approver name out of the resume metadata when present."""
    from fastaiagent.chain.interrupt import _resume_value

    rv = _resume_value.get()
    return rv.metadata.get("approver", "auto") if rv else "auto"


def build_chain() -> Chain:
    chain = Chain("refund-flow", checkpointer=SQLiteCheckpointer())
    chain.add_node(
        "review",
        tool=FunctionTool(name="review_tool", fn=review_step),
        type=NodeType.tool,
        input_mapping={
            "amount": "{{state.amount}}",
            "account": "{{state.account}}",
        },
    )
    return chain


def main() -> None:
    execution_id = "demo-refund-1"

    # First execute — node calls interrupt(); chain suspends.
    chain = build_chain()
    result = chain.execute(
        {"amount": 50_000, "account": "acct-9"},
        execution_id=execution_id,
    )
    print(f"after execute: status={result.status}")
    print(f"  pending: {result.pending_interrupt}")
    print(
        f"  charge_card calls so far: {_charge_count['calls']}"
        "  (0 — node suspended before charging)"
    )

    # Resume — rebuild the chain in a "fresh process"; aresume reads the
    # checkpoint store, claims the pending row, and replays the node
    # with the Resume value in scope.
    chain = build_chain()
    result = run_sync(
        chain.aresume(
            execution_id,
            resume_value=Resume(approved=True, metadata={"approver": "alice"}),
        )
    )
    print(f"\nafter first aresume: status={result.status}")
    print(f"  output: {result.final_state['output']}")
    print(f"  charge_card calls: {_charge_count['calls']}  (1 — @idempotent absorbed the replay)")

    # Second resume on the same execution_id is a deterministic error.
    chain = build_chain()
    try:
        run_sync(chain.aresume(execution_id, resume_value=Resume(approved=True)))
    except AlreadyResumed as exc:
        print(f"\nsecond aresume correctly raised: AlreadyResumed: {exc}")

    print(
        "\nTo render the topology + checkpoint inspector in the Local UI, "
        "register the chain with build_app:\n"
        "    from fastaiagent.ui.server import build_app\n"
        "    app = build_app(runners=[chain])\n"
        "Then visit http://127.0.0.1:7843/workflows/chain/refund-flow "
        "for the topology and /executions/<id> for the checkpoint timeline."
    )


if __name__ == "__main__":
    main()
