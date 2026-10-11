"""Proof for "5 · Between a run that ended and one that died" on docs/durability/durability-boundaries.md.

Offline: a real Agent on the SDK's FunctionModel, a real SQLiteCheckpointer.

A finished run and a run that died right after its last step used to leave the same
rows. The run-end marker tells them apart: completed refuses a resume, failed stays
resumable, and a fork of a finished run branches under a new id and leaves the
original untouched. Pruning never deletes a pause.
"""

import os
from datetime import timedelta

import _agents as A
import _common  # noqa: F401
from _common import checkpointer, heading, rows

from fastaiagent import AlreadyResumed
from fastaiagent._internal.errors import ChainCheckpointError

cp = checkpointer()


def small_bot():
    return A.agent(cp, amount=20)


heading("a finished run refuses to resume")
small_bot().run("Refund order 1042", execution_id="done-1")
print("last row:", rows(cp, "done-1")[-1])
try:
    small_bot().resume("done-1")
except AlreadyResumed as e:
    print("resume →", "AlreadyResumed:", str(e)[:80], "…")

heading("a run that raised stays resumable")
os.environ["PROOF_RAISE"] = "model"   # the provider fails once, after the tool ran
A.charges.clear()
try:
    small_bot().run("Refund order 1042", execution_id="failed-1")
except Exception as e:
    print("run raised:", type(e).__name__, "·", str(e)[:60])
print("last row:", rows(cp, "failed-1")[-1])
r = small_bot().resume("failed-1")
print("resume → status:", r.status, "· charges:", A.charges, "(the tool did not run again)")
os.environ.pop("PROOF_RAISE")

heading("a fork branches a finished run under a new id")
before = len(cp.list("done-1"))
forked = small_bot().fork("done-1", input="Refund order 1042 again, please")
print(f"fork → execution_id={forked.execution_id!r} status={forked.status} · original rows: {before} → {len(cp.list('done-1'))}")
marker = next(c for c in cp.list("done-1") if c.step_type == "run_end")
try:
    small_bot().fork("done-1", checkpoint_id=marker.checkpoint_id)
except ChainCheckpointError as e:
    print("fork from the marker →", "ChainCheckpointError:", str(e)[:70], "…")

heading("prune keeps every pause")
A.agent(cp).run("Refund order 1042", execution_id="paused-1")
deleted = cp.prune(older_than=timedelta(seconds=0))
left = {c.execution_id for c in cp.list("paused-1")} | {c.execution_id for c in cp.list("done-1")}
print(f"prune(older_than=0s) deleted {deleted} rows · still stored: {sorted(left)} · pending: "
      f"{[p.execution_id for p in cp.list_pending_interrupts()]}")
