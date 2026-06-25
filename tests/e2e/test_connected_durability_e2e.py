"""End-to-end quality gate — connected checkpoint durability (WS2).

Runs a real connected SDK chain against the **live** local plane and asserts the
full durability round-trip:

    checkpoint  -> POST /public/v1/checkpoints/ingest      -> synced=1
    restore     <- GET  /public/v1/checkpoints/{id}/latest -> write local -> RESUME

``synced=1`` proves a confirmed 2xx ingest (the outbox marks rows synced only
after a 2xx). Then we simulate local loss with a fresh empty checkpointer, restore
the latest checkpoint **from the plane**, and resume the run to completion — "the
plane serves, the SDK resumes" (no execution on the plane).

NO MOCKS. The plane is the live instance at ``FASTAIAGENT_TARGET``; the key is a
real ``FASTAIAGENT_API_KEY``. The chain is the LLM-free
``seed -> approval(interrupt) -> finalize`` chain from ``test_gate_hitl_suspend``,
so the gate makes zero model calls. Gated by the ``connected_state_plane`` bundle
flag — a 403 skips the gate with a clear setup message.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pytest

from tests.e2e.conftest import require_env, require_platform
from tests.e2e.test_gate_hitl_suspend import _build_chain

pytestmark = pytest.mark.e2e


def test_connected_checkpoint_replicate_restore_resume(
    isolated_local_db: Path, tmp_path: Path
) -> None:
    """Replicate checkpoints to the live plane, then restore + resume from it."""
    require_env()
    require_platform()

    import httpx

    import fastaiagent as fa
    from fastaiagent import Resume
    from fastaiagent._internal.async_utils import run_sync
    from fastaiagent.checkpointers import platform_replica
    from fastaiagent.client import _connection

    fa.connect(
        api_key=os.environ["FASTAIAGENT_API_KEY"],
        target=os.environ["FASTAIAGENT_TARGET"],
    )
    try:
        assert _connection.is_connected, "connect() did not establish a connection"

        # Feature-gate pre-check — a clean skip (not an opaque failure) when the
        # domain isn't entitled to the connected_state_plane bundle flag.
        probe = httpx.post(
            f"{_connection.target}/public/v1/checkpoints/ingest",
            headers=_connection.headers,
            json={"checkpoints": []},
            timeout=10,
        )
        if probe.status_code == 403:
            pytest.skip(
                "connected_state_plane is not enabled for this domain — enable the "
                "Enterprise bundle flag on the plane to run this gate "
                f"(probe HTTP {probe.status_code}: {probe.text[:120]})."
            )
        assert probe.status_code < 400, (
            f"checkpoint ingest probe failed: HTTP {probe.status_code} — {probe.text[:200]}"
        )

        # 1) Run an LLM-free chain that pauses on interrupt() -> checkpoints written.
        ckpt_db1 = str(tmp_path / "ckpt1.db")
        chain, store = _build_chain(ckpt_db1)
        execution_id = f"dur-e2e-{uuid.uuid4().hex[:8]}"
        result = chain.execute(
            {"seed_value": "S1", "amount": 50_000}, execution_id=execution_id
        )
        assert result.status == "paused", f"expected paused, got {result.status!r}"

        # 2) Force a deterministic drain to the live plane (the per-write daemon
        #    kicks also ran), then assert every local checkpoint is synced=1 —
        #    which only happens on a confirmed 2xx ingest.
        deadline = time.monotonic() + 20.0
        rows: list = []
        while time.monotonic() < deadline:
            platform_replica.drain_all_sync()
            rows = store._conn().fetchall(
                "SELECT checkpoint_id, status, synced FROM checkpoints WHERE execution_id = ?",
                (execution_id,),
            )
            if rows and all(r["synced"] == 1 for r in rows):
                break
            time.sleep(0.5)
        assert rows, "no checkpoints were written locally"
        assert all(r["synced"] == 1 for r in rows), f"checkpoints not all synced: {rows}"
        # Non-lossy: the paused run's interrupted checkpoint replicated (not abandoned).
        assert any(r["status"] == "interrupted" for r in rows), rows

        # 3) The plane SERVES the latest checkpoint back (the interrupted one).
        served = platform_replica.fetch_latest_from_plane(execution_id)
        assert served is not None, "plane returned no checkpoint for the execution"
        assert served.execution_id == execution_id
        assert served.status == "interrupted"
        assert served.node_id == "approval"
        assert served.interrupt_reason == "manager_approval"
        assert served.state_snapshot, "served checkpoint lost its chain state"

        # 4) Simulate local loss: a FRESH, empty checkpointer. Restore from plane.
        ckpt_db2 = str(tmp_path / "ckpt2.db")
        chain2, store2 = _build_chain(ckpt_db2)
        assert store2.get_last(execution_id) is None, "fresh store should be empty"

        restored = platform_replica.restore_from_plane(store2, execution_id)
        assert restored is not None and restored.status == "interrupted"
        local_after = store2.get_last(execution_id)
        assert local_after is not None
        assert local_after.checkpoint_id == served.checkpoint_id
        # the pending interrupt was reconstructed so a HITL resume can claim it
        pend = [p for p in store2.list_pending_interrupts() if p.execution_id == execution_id]
        assert len(pend) == 1 and pend[0].node_id == "approval", pend

        # 5) Resume FROM THE PLANE COPY -> the chain completes locally.
        result2 = run_sync(
            chain2.resume(
                execution_id,
                resume_value=Resume(approved=True, metadata={"approver": "alice"}),
            )
        )
        assert result2.status == "completed", (
            f"resume-from-plane did not complete the chain; got {result2.status!r}"
        )
        last_output = result2.final_state.get("output")
        assert isinstance(last_output, dict) and last_output.get("final_decision") is True
    finally:
        platform_replica.drain_all_sync()
        fa.disconnect()
