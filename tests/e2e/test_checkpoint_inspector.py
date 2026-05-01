"""E2E tests for the checkpoint inspector backend surface.

Covers two paths:

1. ``GET /api/executions/{id}`` returns a chronological list of
   checkpoints with all the fields the timeline needs (status,
   state_snapshot, node_input, node_output, interrupt_reason,
   interrupt_context, agent_path, created_at).
2. ``GET /api/executions/{id}/idempotency-cache`` returns the
   ``@idempotent`` rows for that execution.

No mocking — uses the Sprint 1 seed (which writes both checkpoint and
idempotency rows) against a real SQLite DB and the FastAPI app.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

pytest.importorskip("fastapi")
pytest.importorskip("bcrypt")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


# Constants must mirror scripts/seed_ui_sprint1.py.
EXEC_ID = "exec-sprint1-mm-00001"
CHAIN_NAME = "support-triage"


@pytest.fixture
def seeded_sprint1_db(tmp_path: Path) -> Path:
    from scripts.seed_ui_snapshot import seed as seed_base
    from scripts.seed_ui_sprint1 import seed as seed_s1

    db_path = tmp_path / "local.db"
    init_local_db(db_path).close()
    seed_base(db_path)
    seed_s1(db_path)
    return db_path


@pytest.fixture
def no_auth_client(seeded_sprint1_db: Path) -> TestClient:
    app = build_app(db_path=str(seeded_sprint1_db), no_auth=True)
    return TestClient(app)


def test_get_execution_returns_checkpoints_in_order(
    no_auth_client: TestClient,
) -> None:
    r = no_auth_client.get(f"/api/executions/{EXEC_ID}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["execution_id"] == EXEC_ID
    assert body["chain_name"] == CHAIN_NAME
    cps = body["checkpoints"]
    assert len(cps) == 2

    # The seed lays them down in step order: completed → interrupted.
    assert cps[0]["node_id"] == "research"
    assert cps[0]["status"] == "completed"
    assert cps[0]["state_snapshot"]["query"] == "refund policy"

    assert cps[1]["node_id"] == "approval"
    assert cps[1]["status"] == "interrupted"
    assert cps[1]["interrupt_reason"] == "manager_approval"
    assert cps[1]["interrupt_context"]["context"]["amount"] == 500


def test_idempotency_cache_lists_every_cached_call(
    no_auth_client: TestClient,
) -> None:
    r = no_auth_client.get(
        f"/api/executions/{EXEC_ID}/idempotency-cache"
    )
    assert r.status_code == 200, r.text
    body = r.json()
    keys = {item["function_key"] for item in body["items"]}
    assert keys == {
        "charge_customer:cust_42:500",
        "send_notification:alice@x.io:refund",
    }
    # Result was JSON-decoded back to a dict on the way out.
    charge = next(
        item for item in body["items"]
        if item["function_key"].startswith("charge_customer")
    )
    assert charge["result"] == {"charge_id": "ch_abc"}


def test_idempotency_cache_empty_for_unknown_execution(
    no_auth_client: TestClient,
) -> None:
    r = no_auth_client.get(
        "/api/executions/does-not-exist/idempotency-cache"
    )
    assert r.status_code == 200
    assert r.json()["count"] == 0


def test_pending_interrupt_row_lights_up_for_seeded_execution(
    no_auth_client: TestClient,
) -> None:
    """Sprint 1 seed writes the matching ``pending_interrupts`` row so the
    Approvals page knows which execution is live."""
    r = no_auth_client.get("/api/pending-interrupts")
    assert r.status_code == 200
    items = r.json()["items"]
    by_exec = {it["execution_id"]: it for it in items}
    assert EXEC_ID in by_exec
    pending = by_exec[EXEC_ID]
    assert pending["chain_name"] == CHAIN_NAME
    assert pending["node_id"] == "approval"
    assert pending["reason"] == "manager_approval"
