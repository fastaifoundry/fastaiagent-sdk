"""Proof for "1 · Between one step and the next" on docs/durability/durability-boundaries.md.

Offline: a real Agent on the SDK's FunctionModel, a real SQLiteCheckpointer on a scratch file.

A durable run is rows. One is written before each model call, one before each tool
runs, and one when the run ends. A pause is one more row, status interrupted, written
in the same transaction as the pending row the approvals UI reads — and a paused run
has no end row, because a pause is not an ending.
"""

import _agents as A
import _common
from _common import checkpointer, heading, rows

cp = checkpointer()

heading("a run that completes: one small refund")
r = A.agent(cp, amount=20).run("Refund order 1042", execution_id="run-small")
print("status:", r.status, "· output:", repr(r.output))
for line in rows(cp, "run-small"):
    print(" ", line)

heading("a run that pauses: a large refund")
r = A.agent(cp).run("Refund order 1042", execution_id="run-large")
print("status:", r.status, "· pending_interrupt:", {k: r.pending_interrupt[k] for k in ("reason", "node_id", "agent_path")})
for line in rows(cp, "run-large"):
    print(" ", line)
pending = cp.list_pending_interrupts()
print("pending_interrupts rows:", [(p.execution_id, p.reason, p.context) for p in pending])
print("run-end rows for the paused run:", [c.node_id for c in cp.list("run-large") if c.step_type == "run_end"])
print("checkpoint file:", _common.CKPT_DB.name, "·", _common.CKPT_DB.stat().st_size // 1024, "KB")
