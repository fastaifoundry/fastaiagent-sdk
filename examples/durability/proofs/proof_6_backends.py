"""Proof for "6 · Between a laptop and a fleet, and the plane" on docs/durability/durability-boundaries.md.

Offline: SQLite always; Postgres when PG_TEST_DSN is set (scripts/dev_backends.sh up).
No plane is contacted.

The same pause, claim and resume on both backends: five racing resumers, one winner.
And the outbox a connected checkpointer keeps: rows wait un-acked until the plane
confirms them, and nothing leaves the machine while disconnected.
"""

import os
import threading

import _agents as A
from _common import checkpointer, heading

from fastaiagent import AlreadyResumed, Resume
from fastaiagent.checkpointers.protocol import Checkpointer, ReplicatedCheckpointer

backends = {"sqlite": checkpointer()}
dsn = os.environ.get("PG_TEST_DSN")
if dsn:
    from fastaiagent.checkpointers import PostgresCheckpointer

    backends["postgres"] = PostgresCheckpointer(dsn, schema="proof_durability")
else:
    print("PG_TEST_DSN not set — the Postgres half is skipped (scripts/dev_backends.sh up)")

for name, cp in backends.items():
    heading(f"{name}: pause, race, claim")
    cp.setup()
    exec_id = f"race-{name}-{os.getpid()}"
    A.agent(cp).run("Refund order 1042", execution_id=exec_id)
    outcomes: list[str] = []
    lock = threading.Lock()

    def resumer(cp=cp, exec_id=exec_id):
        try:
            A.agent(cp).resume(exec_id, resume_value=Resume(approved=True))
            out = "completed"
        except AlreadyResumed:
            out = "AlreadyResumed"
        with lock:
            outcomes.append(out)

    threads = [threading.Thread(target=resumer) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"  protocol: Checkpointer={isinstance(cp, Checkpointer)} Replicated={isinstance(cp, ReplicatedCheckpointer)}")
    print(f"  outcomes: {sorted(outcomes)}")
    print(f"  claim   : {'DELETE … RETURNING' if name == 'postgres' else 'SELECT + DELETE under one lock'}")

    heading(f"{name}: the outbox, disconnected")
    waiting = cp.fetch_unsynced(limit=100)
    mine = [r for r in waiting if r.get("execution_id") == exec_id]
    print(f"  un-acked rows for this run: {len(mine)} (synced only after the plane answers 2xx; none sent — not connected)")
    cp.mark_synced([r["checkpoint_id"] for r in mine])
    print(f"  after mark_synced: {len([r for r in cp.fetch_unsynced(limit=100) if r.get('execution_id') == exec_id])} waiting")
    if name == "postgres":
        cp.delete_execution(exec_id)
