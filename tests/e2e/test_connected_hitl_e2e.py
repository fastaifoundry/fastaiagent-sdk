"""End-to-end quality gate — connected HITL observer reporting (WS1).

Runs a real connected SDK chain against the **live** local plane and asserts the
two-way round-trip:

    pause  (interrupt())          -> POST /public/v1/hitl/events  -> synced=1
    resolution (chain.resume())   -> POST /public/v1/hitl/events  -> synced=1

``synced=1`` is the proof: :class:`HitlEventExporter` marks a local outbox row
synced **only after a confirmed 2xx** from the plane, so a flipped flag means the
plane really ingested the event.

NO MOCKS. The plane is the live instance at ``FASTAIAGENT_TARGET`` and the key is
a real ``FASTAIAGENT_API_KEY``. The chain is the same LLM-free
``seed -> approval(interrupt) -> finalize`` chain as ``test_gate_hitl_suspend``,
so the gate makes **zero** model calls.

The connected-HITL ingest is gated by the ``connected_state_plane`` Enterprise
bundle flag. If the domain isn't entitled the plane returns 403 — this gate then
**skips** with a clear setup message instead of failing opaquely.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.conftest import require_env, require_platform
from tests.e2e.test_gate_hitl_suspend import _build_chain

pytestmark = pytest.mark.e2e


def _read_event(db_path: Path, run_id: str, event_type: str) -> dict[str, Any] | None:
    from fastaiagent._internal.storage import SQLiteHelper

    db = SQLiteHelper(str(db_path))
    try:
        return db.fetchone(
            "SELECT * FROM hitl_events WHERE run_id = ? AND event_type = ?",
            (run_id, event_type),
        )
    finally:
        db.close()


def _drain_until_synced(
    db_path: Path, run_id: str, event_type: str, timeout: float = 20.0
) -> dict[str, Any]:
    """Drive the real drain to the live plane and poll the local row until synced=1.

    Calling the exporter's ``export()`` synchronously exercises the exact same
    drain path as the per-emit daemon-thread kick, but deterministically: it
    POSTs to the live ``/public/v1/hitl/events`` and marks the row ``synced=1``
    only on a confirmed 2xx.
    """
    from fastaiagent.trace.hitl_export import get_hitl_exporter

    deadline = time.monotonic() + timeout
    row: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        get_hitl_exporter().export([])  # synchronous drain against the live plane
        row = _read_event(db_path, run_id, event_type)
        if row is not None and row["synced"] == 1:
            return row
        time.sleep(0.5)
    raise AssertionError(
        f"HITL {event_type!r} event for run {run_id!r} was not synced within "
        f"{timeout}s (row={row}). The live plane never returned a 2xx."
    )


def test_connected_hitl_pause_and_resolution_roundtrip(
    isolated_local_db: Path, tmp_path: Path
) -> None:
    """Pause and resolution both round-trip to the LIVE plane (synced=1)."""
    require_env()
    require_platform()

    import httpx

    import fastaiagent as fa
    from fastaiagent import Resume
    from fastaiagent._internal.async_utils import run_sync
    from fastaiagent.client import _connection
    from fastaiagent.trace.hitl_export import get_hitl_exporter

    local_db = isolated_local_db  # temp local.db; project pinned to "test-proj"

    # 1) Connect to the LIVE plane (real key, real endpoint, no mocks).
    fa.connect(
        api_key=os.environ["FASTAIAGENT_API_KEY"],
        target=os.environ["FASTAIAGENT_TARGET"],
    )
    try:
        assert _connection.is_connected, "connect() did not establish a connection"

        # Feature-gate pre-check — a clean skip (not an opaque failure) when the
        # domain isn't entitled to the connected_state_plane bundle flag.
        probe = httpx.post(
            f"{_connection.target}/public/v1/hitl/events",
            headers=_connection.headers,
            json={"events": []},
            timeout=10,
        )
        if probe.status_code == 403:
            pytest.skip(
                "connected_state_plane is not enabled for this domain — enable the "
                "Enterprise bundle flag on the plane to run this gate "
                f"(probe HTTP {probe.status_code}: {probe.text[:120]})."
            )
        assert probe.status_code < 400, (
            f"HITL ingest probe failed: HTTP {probe.status_code} — {probe.text[:200]}"
        )

        # 2) Run an LLM-free chain that pauses on interrupt().
        ckpt_db = str(tmp_path / "ckpt.db")
        chain, _store = _build_chain(ckpt_db)
        execution_id = f"hitl-e2e-{uuid.uuid4().hex[:8]}"

        result = chain.execute(
            {"seed_value": "S1", "amount": 50_000},
            execution_id=execution_id,
        )
        assert result.status == "paused", (
            f"expected a paused chain, got status={result.status!r}"
        )

        # 3) The pause event round-tripped to the live plane (synced=1).
        pause_row = _drain_until_synced(local_db, execution_id, "paused")
        assert pause_row["event_type"] == "paused"
        assert pause_row["kind"] == "interrupt"
        assert pause_row["chain_id"] == "hitl-suspend-gate"
        assert pause_row["node"] == "approval"
        assert pause_row["reason"] == "manager_approval"
        # status/resolver are resolution-only — never set on a pause.
        assert pause_row["status"] is None
        assert pause_row["resolver"] is None

        # 4) Resume (the customer-side approval) with a resolver identity.
        resolved = run_sync(
            chain.resume(
                execution_id,
                resume_value=Resume(
                    approved=True, metadata={"resolver": "alice@acme.com"}
                ),
            )
        )
        assert resolved.status == "completed", (
            f"resume did not complete the chain; got status={resolved.status!r}"
        )

        # 5) The resolution event round-tripped (synced=1, approved, resolver).
        res_row = _drain_until_synced(local_db, execution_id, "resolved")
        assert res_row["event_type"] == "resolved"
        assert res_row["status"] == "approved"
        assert res_row["resolver"] == "alice@acme.com"
        assert res_row["node"] == "approval"
        assert res_row["chain_id"] == "hitl-suspend-gate"
    finally:
        get_hitl_exporter().shutdown()
        fa.disconnect()
