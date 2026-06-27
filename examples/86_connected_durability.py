"""Example 86 — Connected checkpoint durability (WS2).

When ``connect()``-ed to an Enterprise control plane, the SDK replicates its
local checkpoints to the plane (the **managed durable copy**) so a run can be
restored and resumed even if the local store is lost — "the plane serves, the SDK
resumes" (no agent code ever runs on the plane).

This demo:

* runs an LLM-free chain that pauses on ``interrupt()`` (writing checkpoints);
* drains the outbox to ``POST /public/v1/checkpoints/ingest`` (``synced=1`` proves
  a confirmed 2xx — and the paused run's checkpoint is **non-lossy**, never
  abandoned);
* simulates local loss with a **fresh, empty** checkpointer, restores the latest
  checkpoint from ``GET /public/v1/checkpoints/{id}/latest``, and **resumes to
  completion** from the plane copy.

See docs/durability/connected-checkpoints.md for the console run-health view.

Usage:
    export FASTAIAGENT_API_KEY=fa_k_...          # a connected_state_plane domain key
    export FASTAIAGENT_TARGET=http://localhost:20001
    python examples/86_connected_durability.py

Expected output (snapshot — real run against a local plane on :20001):
    connected to http://localhost:20001
    chain paused: status=paused (checkpoints written locally)
    replicated: 2 checkpoint(s) synced=1 (incl. the interrupted one — non-lossy)
    plane served latest: status=interrupted node=approval
    restored into a FRESH checkpointer; resuming from the plane copy...
    resumed from plane: status=completed final_decision=True
    done — durability round-trip (replicate -> restore -> resume) complete.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import fastaiagent as fa
from fastaiagent import Chain, FunctionTool, Resume, SQLiteCheckpointer
from fastaiagent._internal.async_utils import run_sync
from fastaiagent.chain.interrupt import interrupt
from fastaiagent.chain.node import NodeType
from fastaiagent.checkpointers import platform_replica
from fastaiagent.client import _connection


def _seed(value: str) -> dict[str, Any]:
    return {"seeded": value, "step_seed_done": True}


def _approval(amount: str) -> dict[str, Any]:
    if int(amount) > 1000:
        decision = interrupt(reason="manager_approval", context={"amount": int(amount)})
        return {"approved": decision.approved, "step_approval_done": True}
    return {"approved": True, "auto": True, "step_approval_done": True}


def _finalize(approved: str, seeded: str) -> dict[str, Any]:
    return {"final_seed": seeded, "final_decision": str(approved).lower() == "true"}


def _build_chain(ckpt_db: str) -> tuple[Chain, SQLiteCheckpointer]:
    store = SQLiteCheckpointer(db_path=ckpt_db)
    chain = Chain("connected-durability-demo", checkpoint_enabled=True, checkpointer=store)
    chain.add_node(
        "seed",
        tool=FunctionTool(name="seed", fn=_seed),
        type=NodeType.tool,
        input_mapping={"value": "{{state.seed_value}}"},
    )
    chain.add_node(
        "approval",
        tool=FunctionTool(name="approval", fn=_approval),
        type=NodeType.tool,
        input_mapping={"amount": "{{state.amount}}"},
    )
    chain.add_node(
        "finalize",
        tool=FunctionTool(name="finalize", fn=_finalize),
        type=NodeType.tool,
        input_mapping={"approved": "{{state.output.approved}}", "seeded": "{{state.seed_value}}"},
    )
    chain.connect("seed", "approval")
    chain.connect("approval", "finalize")
    return chain, store


def main() -> int:
    api_key = os.environ.get("FASTAIAGENT_API_KEY", "")
    target = os.environ.get("FASTAIAGENT_TARGET", "http://localhost:20001")
    if not api_key:
        print("Skipping: set FASTAIAGENT_API_KEY + FASTAIAGENT_TARGET (connected_state_plane key)")
        return 1

    fa.connect(api_key=api_key, target=target)
    print(f"connected to {target}")
    try:
        import httpx

        probe = httpx.post(
            f"{target}/public/v1/checkpoints/ingest",
            headers=_connection.headers,
            json={"checkpoints": []},
            timeout=10,
        )
        if probe.status_code == 403:
            print("Skipping: connected_state_plane is not enabled for this domain (HTTP 403).")
            return 1

        execution_id = f"dur-demo-{uuid.uuid4().hex[:8]}"
        chain, store = _build_chain(f".fastaiagent/dur-demo-1-{execution_id}.db")
        result = chain.execute({"seed_value": "S1", "amount": 50_000}, execution_id=execution_id)
        print(f"chain paused: status={result.status} (checkpoints written locally)")

        # Drain the outbox until every checkpoint is confirmed synced.
        deadline = time.monotonic() + 15.0
        rows: list = []
        while time.monotonic() < deadline:
            platform_replica.drain_all_sync()
            rows = store._conn().fetchall(
                "SELECT status, synced FROM checkpoints WHERE execution_id = ?", (execution_id,)
            )
            if rows and all(r["synced"] == 1 for r in rows):
                break
            time.sleep(0.5)
        n_synced = sum(1 for r in rows if r["synced"] == 1)
        has_interrupted = any(r["status"] == "interrupted" for r in rows)
        print(
            f"replicated: {n_synced} checkpoint(s) synced=1"
            + (" (incl. the interrupted one — non-lossy)" if has_interrupted else "")
        )

        served = platform_replica.fetch_latest_from_plane(execution_id)
        if served is None:
            print("plane returned no checkpoint — aborting.")
            return 1
        print(f"plane served latest: status={served.status} node={served.node_id}")

        # Simulate local loss: a brand-new, empty checkpointer.
        chain2, store2 = _build_chain(f".fastaiagent/dur-demo-2-{execution_id}.db")
        platform_replica.restore_from_plane(store2, execution_id)
        print("restored into a FRESH checkpointer; resuming from the plane copy...")

        resumed = run_sync(
            chain2.resume(
                execution_id, resume_value=Resume(approved=True, metadata={"approver": "alice"})
            )
        )
        final = (resumed.final_state or {}).get("output", {})
        print(
            f"resumed from plane: status={resumed.status} "
            f"final_decision={final.get('final_decision')}"
        )
        print("done — durability round-trip (replicate -> restore -> resume) complete.")
        return 0
    finally:
        platform_replica.drain_all_sync()
        fa.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
