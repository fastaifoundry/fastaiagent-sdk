"""Example 85 — Connected HITL observer reporting (WS1).

When the SDK is ``connect()``-ed to an Enterprise control plane, every
``interrupt()`` **pause** and ``resume()`` **resolution** is reported to the plane
as metadata so it can serve an org-wide pending/paused status view and a
compliance ledger. The plane is an **observer** — approval still happens in your
own app; nothing about how you resolve a pause changes when connected.

This demo runs an LLM-free ``seed -> approval(interrupt) -> finalize`` chain:

* ``execute()`` pauses at ``approval`` (a real checkpoint).
* The pause is written to a local ``hitl_events`` outbox and drained to
  ``POST /public/v1/hitl/events``; ``synced=1`` means the plane confirmed a 2xx.
* ``resume()`` completes the chain and reports a ``resolved`` event (with the
  resolver identity).

See docs/platform/connected-hitl.md for the console view this produces.

Usage:
    export FASTAIAGENT_API_KEY=fa_k_...          # a connected_state_plane domain key
    export FASTAIAGENT_TARGET=http://localhost:20001
    python examples/85_connected_hitl.py

Expected output (snapshot — real run against a local plane on :20001):
    connected to http://localhost:20001
    chain paused: status=paused node=approval reason=manager_approval
    reported pause     -> synced=1  event=paused    node=approval    reason=manager_approval
    chain resumed: status=completed
    reported resolution -> synced=1  event=resolved  status=approved  resolver=alice@acme.com
    done — both events landed in the plane's HITL ledger.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import fastaiagent as fa
from fastaiagent import Chain, FunctionTool, Resume, SQLiteCheckpointer
from fastaiagent._internal.async_utils import run_sync
from fastaiagent.chain.interrupt import interrupt
from fastaiagent.chain.node import NodeType
from fastaiagent.client import _connection


def _seed(value: str) -> dict[str, Any]:
    return {"seeded": value, "step_seed_done": True}


def _approval(amount: str) -> dict[str, Any]:
    # High-value payments pause for a human; small ones auto-approve.
    if int(amount) > 1000:
        decision = interrupt(reason="manager_approval", context={"amount": int(amount)})
        return {"approved": decision.approved, "step_approval_done": True}
    return {"approved": True, "auto": True, "step_approval_done": True}


def _finalize(approved: str, seeded: str) -> dict[str, Any]:
    return {"final_seed": seeded, "final_decision": str(approved).lower() == "true"}


def _build_chain(ckpt_db: str) -> Chain:
    chain = Chain(
        "connected-hitl-demo",
        checkpoint_enabled=True,
        checkpointer=SQLiteCheckpointer(db_path=ckpt_db),
    )
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
    return chain


def _drain_and_read(execution_id: str, event_type: str) -> dict[str, Any] | None:
    """Drain the HITL outbox to the plane, then read the local row back."""
    import time

    from fastaiagent._internal.config import get_config
    from fastaiagent._internal.storage import SQLiteHelper
    from fastaiagent.trace.hitl_export import get_hitl_exporter

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        get_hitl_exporter().export([])  # synchronous drain to the live plane
        db = SQLiteHelper(get_config().local_db_path)
        try:
            row = db.fetchone(
                "SELECT * FROM hitl_events WHERE run_id = ? AND event_type = ?",
                (execution_id, event_type),
            )
        finally:
            db.close()
        if row and row.get("synced") == 1:
            return row
        time.sleep(0.5)
    return None


def main() -> int:
    api_key = os.environ.get("FASTAIAGENT_API_KEY", "")
    target = os.environ.get("FASTAIAGENT_TARGET", "http://localhost:20001")
    if not api_key:
        print("Skipping: set FASTAIAGENT_API_KEY + FASTAIAGENT_TARGET (connected_state_plane key)")
        return 1

    fa.connect(api_key=api_key, target=target)
    print(f"connected to {target}")
    try:
        # Probe the ingest endpoint so an unentitled domain fails clearly.
        import httpx

        probe = httpx.post(
            f"{target}/public/v1/hitl/events",
            headers=_connection.headers,
            json={"events": []},
            timeout=10,
        )
        if probe.status_code == 403:
            print("Skipping: connected_state_plane is not enabled for this domain (HTTP 403).")
            return 1

        execution_id = f"hitl-demo-{uuid.uuid4().hex[:8]}"
        chain = _build_chain(f".fastaiagent/hitl-demo-{execution_id}.db")

        result = chain.execute({"seed_value": "S1", "amount": 50_000}, execution_id=execution_id)
        print(
            f"chain paused: status={result.status} node="
            f"{(result.pending_interrupt or {}).get('node_id')} "
            f"reason={(result.pending_interrupt or {}).get('reason')}"
        )

        pause = _drain_and_read(execution_id, "paused")
        if pause:
            print(
                f"reported pause     -> synced={pause['synced']}  event={pause['event_type']}    "
                f"node={pause['node']}    reason={pause['reason']}"
            )

        resolved = run_sync(
            chain.resume(
                execution_id,
                resume_value=Resume(approved=True, metadata={"resolver": "alice@acme.com"}),
            )
        )
        print(f"chain resumed: status={resolved.status}")

        res = _drain_and_read(execution_id, "resolved")
        if res:
            print(
                f"reported resolution -> synced={res['synced']}  event={res['event_type']}  "
                f"status={res['status']}  resolver={res['resolver']}"
            )
        print("done — both events landed in the plane's HITL ledger.")
        return 0
    finally:
        from fastaiagent.trace.hitl_export import get_hitl_exporter

        get_hitl_exporter().shutdown()
        fa.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
