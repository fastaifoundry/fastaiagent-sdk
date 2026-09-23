"""End-to-end quality gate — the calling application resolves a policy approval (1.74.0).

No mocks. A real SDK and a real ``gpt-4o-mini`` agent against the LIVE local
plane. This is the reproduction from the plane's handoff
(``fastaiagent-enterprise/docs/Plane_Handoff_Approvals_2026-09.md`` §2 and
"Verifying"), kept as a gate:

    an approval policy covers the tool      (authored through the console API)
    arun()                                  -> paused, the pause carries the arguments
    aresume(Resume(approved=False, resolver))
                                            -> the tool NEVER runs, the model is told
    the plane's HITL ledger                 -> kind=approval, resolved/rejected, the resolver
    the plane's pending run                 -> closed as rejected, by that resolver

On 1.73.0 the tool ran and the model reported a successful transfer.

The last assertion (the pending run closed with the app's resolver) is the
plane's half of the same change — the approvals-observer ledger, plane branch
``fix/approvals-observer-ledger``. A plane without it leaves the row ``pending``.

The policy covers a tool name unique to this run, so a policy left behind by an
earlier run (a policy with approval history cannot be deleted, only deactivated)
can never gate anything else.

Setup is the guardrail-actions gate's (one persistent lab domain):

    E2E_PLANE_EMAIL / E2E_PLANE_PASSWORD  a domain-admin on the local plane
    FASTAIAGENT_TARGET                    http://localhost:20001
    FASTAIAGENT_API_KEY                   a key minted in that domain (agent:write)

Run:

    zsh -lc 'FASTAIAGENT_TARGET=http://localhost:20001 FASTAIAGENT_API_KEY=fa_k_... \
      E2E_PLANE_EMAIL=... E2E_PLANE_PASSWORD=... \
      .venv/bin/python -m pytest tests/e2e/test_connected_approvals_e2e.py -v -m e2e'
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.conftest import require_env, require_platform

pytestmark = pytest.mark.e2e

RUN = uuid.uuid4().hex[:8]
TOOL = f"transfer_funds_{RUN}"


def _admin(target: str) -> tuple[Any, dict[str, str], str]:
    """A domain-admin session on the lab domain, the way an operator authors policy."""
    import httpx

    email = os.environ.get("E2E_PLANE_EMAIL")
    password = os.environ.get("E2E_PLANE_PASSWORD")
    if not (email and password):
        pytest.skip(
            "E2E_PLANE_EMAIL / E2E_PLANE_PASSWORD not set — this gate authors an "
            "approval policy through the console API, which needs a domain admin."
        )
    client = httpx.Client(base_url=target, timeout=30)
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    if resp.status_code != 200:
        pytest.skip(f"plane login failed ({resp.status_code}) — is the local plane running?")
    headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
    domain_id = os.environ.get("E2E_PLANE_DOMAIN_ID")
    if not domain_id:
        domains = client.get("/api/v1/users/me/domains", headers=headers).json()
        lab = next((d for d in domains if d["name"] == "Guardrail Actions Lab"), None)
        if lab is None:
            pytest.skip(
                "no 'Guardrail Actions Lab' domain on this plane — create it (or set "
                "E2E_PLANE_DOMAIN_ID) so the gate does not author policy into a shared domain."
            )
        domain_id = lab["id"]
    return client, headers, domain_id


@pytest.fixture
def lab(isolated_local_db: Path) -> Any:
    require_env()
    require_platform()

    import httpx

    import fastaiagent as fa

    target = os.environ["FASTAIAGENT_TARGET"]
    key = os.environ["FASTAIAGENT_API_KEY"]
    probe = httpx.get(f"{target}/public/v1/policy", headers={"X-API-Key": key}, timeout=30)
    if probe.status_code == 403:
        pytest.skip("connected_state_plane not enabled for this domain.")
    assert probe.status_code == 200, f"/public/v1/policy returned {probe.status_code}"

    client, headers, domain_id = _admin(target)
    created = client.post(
        "/api/v1/approval-policies",
        params={"domain_id": domain_id},
        headers=headers,
        json={
            "name": f"sdk-approvals-e2e-{RUN}",
            "description": "SDK approvals e2e (safe to delete)",
            "tool_pattern": TOOL,
            "condition_type": "always",
            "timeout_minutes": 1,
        },
    )
    assert created.status_code == 201, created.text
    policy_id = created.json()["id"]
    fa.connect(api_key=key, target=target)
    try:
        yield client, headers, domain_id
    finally:
        fa.disconnect()
        gone = client.delete(f"/api/v1/approval-policies/{policy_id}", headers=headers)
        if gone.status_code == 409:  # it has approval history: deactivate instead
            client.put(
                f"/api/v1/approval-policies/{policy_id}",
                headers=headers,
                json={"is_active": False},
            )


def test_a_rejected_approval_never_runs_the_tool_and_is_recorded(lab: Any, tmp_path: Path) -> None:
    import fastaiagent as fa
    from fastaiagent import Agent, FunctionTool, LLMClient
    from fastaiagent.chain.interrupt import Resume
    from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer
    from fastaiagent.trace.hitl_export import get_hitl_exporter

    client, headers, domain_id = lab
    ran: list[dict[str, Any]] = []

    def transfer(amount: int, to: str) -> str:
        ran.append({"amount": amount, "to": to})
        return f"Transferred ${amount} to {to}."

    agent = Agent(
        name=f"approvals-e2e-{RUN}",
        system_prompt=(
            f"You are a banking assistant. To move money, call {TOOL}(amount, to). "
            "After the tool returns, tell the user in one sentence exactly what it said."
        ),
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        tools=[FunctionTool(name=TOOL, fn=transfer)],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )
    agent.push()  # the platform agent id is what /policy/decide gates on
    assert agent.agent_id
    fa.refresh_policy()

    run_id = f"approvals-e2e-{RUN}"
    paused = asyncio.run(agent.arun("Transfer $500 to Bob.", execution_id=run_id))
    assert paused.status == "paused", paused
    assert paused.pending_interrupt["context"]["tool_input"] == {"amount": 500, "to": "Bob"}

    resolver = f"e2e-{RUN}@app.example"
    final = asyncio.run(
        agent.aresume(run_id, resume_value=Resume(approved=False, metadata={"resolver": resolver}))
    )

    assert final.status == "completed", final
    assert ran == [], f"a rejected approval executed the tool: {ran}"
    assert "transferred $500" not in final.output.lower(), final.output

    get_hitl_exporter().export([])
    ledger = client.get(
        "/api/v1/hitl/events",
        params={"domain_id": domain_id, "run_id": run_id},
        headers=headers,
    )
    assert ledger.status_code == 200, ledger.text
    rows = [(e["event_type"], e["kind"], e["status"], e["resolver"]) for e in ledger.json()]
    assert rows == [
        ("paused", "approval", None, None),
        ("resolved", "approval", "rejected", resolver),
    ], rows

    pending = client.get(
        "/api/v1/pending-runs", params={"domain_id": domain_id}, headers=headers
    ).json()
    mine = [p for p in pending if p["run_id"] == run_id]
    assert [(p["status"], p["resolved_by"]) for p in mine] == [("rejected", resolver)], mine
