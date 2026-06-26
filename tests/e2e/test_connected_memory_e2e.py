"""End-to-end quality gate — connected central memory read (WS3).

Asserts the SDK read side of central governed memory against the **live** plane:
a connected ``PlaneFactBlock`` reads a curated, human-approved fact back from
``GET /public/v1/memory/facts`` and injects it; a redaction removes it org-wide
on the next read; and when disconnected the block is a strict no-op.

NO MOCKS. The plane is the live instance at ``FASTAIAGENT_TARGET``. Per D5
(central-extraction-only) the SDK has **no fact-push path**, so the fact must be
produced on the plane. This gate seeds plane state via the real Enterprise API
(create project → push agent definition → create an approved fact), then asserts
the SDK READ — the SDK's actual WS3 responsibility. Central extraction-from-traces
+ the annotation queue are exercised Enterprise-side; a manually-created fact is
auto-approved and served by the same endpoint, so it exercises the same read path.

Seeding + redaction need a domain-admin session, so the gate reads
``E2E_PLANE_EMAIL`` / ``E2E_PLANE_PASSWORD`` from the env and skips cleanly when
they're absent. Gated by ``connected_state_plane`` (403 → skip with a setup note).
"""

from __future__ import annotations

import os
import uuid

import pytest

from tests.e2e.conftest import require_env, require_platform

pytestmark = pytest.mark.e2e


def test_connected_memory_facts_read_and_redact(isolated_local_db) -> None:
    require_env()
    require_platform()

    email = os.environ.get("E2E_PLANE_EMAIL")
    password = os.environ.get("E2E_PLANE_PASSWORD")
    if not (email and password):
        pytest.skip(
            "set E2E_PLANE_EMAIL / E2E_PLANE_PASSWORD (a domain-admin on the local "
            "plane) to run the WS3 memory gate — seeding + redaction need a session."
        )

    import httpx

    import fastaiagent as fa
    from fastaiagent.agent.memory_blocks import PlaneFactBlock
    from fastaiagent.client import _connection

    base = os.environ["FASTAIAGENT_TARGET"].rstrip("/")
    http = httpx.Client(timeout=30)

    # --- plane fixture setup (real Enterprise API) -------------------------
    tok = http.post(
        f"{base}/api/v1/auth/login", json={"email": email, "password": password}
    ).json()["access_token"]
    jwt = {"Authorization": f"Bearer {tok}"}

    domains = http.get(f"{base}/api/v1/users/me/domains", headers=jwt).json()
    assert domains, "login user belongs to no domain"
    domain_id = domains[0]["id"]

    # A project must exist in the domain so the SDK agent push can resolve one.
    # 201 = created; 409 = already exists from a prior run — both satisfy that.
    # The agent push resolves the project itself, so we don't need its id here.
    pr = http.post(
        f"{base}/api/v1/domains/{domain_id}/projects",
        headers=jwt,
        json={"name": "WS3 E2E", "slug": "ws3-e2e"},
    )
    assert pr.status_code in (200, 201, 409), (
        f"project setup failed: HTTP {pr.status_code} {pr.text[:160]}"
    )

    # Free an API-key slot (the free tier caps keys) by dropping any leftover
    # test keys from prior runs, then mint a fresh write-scoped key.
    lk = http.get(f"{base}/api/v1/api-keys", headers=jwt)
    if lk.status_code == 200:
        keys = lk.json()
        keys = keys if isinstance(keys, list) else keys.get("keys", [])
        for k in keys:
            if str(k.get("name", "")).startswith("ws3-e2e") and k.get("id"):
                http.delete(f"{base}/api/v1/api-keys/{k['id']}", headers=jwt)

    # An API key with agent:write (to push the agent definition) + agent:execute
    # (the read needs only a valid domain key). The SDK connects with this key.
    kr = http.post(
        f"{base}/api/v1/api-keys",
        headers=jwt,
        json={
            "name": f"ws3-e2e-{uuid.uuid4().hex[:6]}",
            "permissions": ["read", "write", "execute"],
            "domain_id": domain_id,
            "scopes": ["agent:write", "agent:execute"],
        },
    )
    assert kr.status_code == 201, f"key mint failed: {kr.status_code} {kr.text[:160]}"
    api_key = kr.json()["key"]
    api_key_id = kr.json().get("id")
    key_hdr = {"X-API-Key": api_key}

    # Push the agent DEFINITION (no execution — the plane runs no agent code, §1).
    ar = http.post(
        f"{base}/public/v1/sdk/agents",
        headers=key_hdr,
        json={"name": "ws3-mem-e2e", "system_prompt": "Support agent.", "agent_type": "single"},
    )
    assert ar.status_code in (200, 201), f"agent push failed: {ar.status_code} {ar.text[:200]}"
    agent_id = ar.json().get("id") or ar.json().get("agent_id")
    assert agent_id

    # 403 pre-check — clean skip when the domain isn't entitled to the bundle flag.
    probe = http.get(
        f"{base}/public/v1/memory/facts", headers=key_hdr, params={"agent_id": agent_id}
    )
    if probe.status_code == 403:
        pytest.skip(
            "connected_state_plane is not enabled for this domain — enable the "
            f"Enterprise bundle flag to run this gate (probe HTTP {probe.status_code})."
        )
    assert probe.status_code == 200, f"facts probe failed: {probe.status_code} {probe.text[:200]}"

    # Seed a curated, approved fact (manual create → auto-approved + active).
    fact = f"The customer's preferred contact channel is email. [{uuid.uuid4().hex[:8]}]"
    mr = http.post(
        f"{base}/api/v1/agents/{agent_id}/memories",
        headers=jwt,
        json={"content": fact, "category": "preferences", "importance": 0.9},
    )
    assert mr.status_code == 201, f"memory create failed: {mr.status_code} {mr.text[:200]}"
    assert mr.json().get("curation_status") == "approved"
    memory_id = mr.json()["id"]

    # --- SDK assertion: the block reads the curated fact from the live plane ---
    fa.connect(api_key=api_key, target=base)
    try:
        assert _connection.is_connected

        block = PlaneFactBlock(agent_id=agent_id, query_conditioned=False)
        msgs = block.render("what channel does the customer prefer?")
        assert len(msgs) == 1, f"expected one system message, got {msgs!r}"
        assert fact in msgs[0].content, f"curated fact not injected: {msgs[0].content!r}"
        assert msgs[0].content.startswith(f"Curated facts (agent:{agent_id})")

        # Central redaction removes the fact org-wide → the next read drops it.
        rd = http.delete(
            f"{base}/api/v1/agents/{agent_id}/memories/{memory_id}/redact",
            headers=jwt,
            params={"reason": "e2e cleanup"},
        )
        assert rd.status_code == 200, f"redact failed: {rd.status_code} {rd.text[:160]}"

        # A fresh block (no cache) sees the redaction: no facts → empty render.
        after = PlaneFactBlock(agent_id=agent_id, query_conditioned=False)
        assert after.render("what channel does the customer prefer?") == []
    finally:
        fa.disconnect()
        # Self-clean the minted key so re-runs stay under the tier's key cap.
        if api_key_id:
            http.delete(f"{base}/api/v1/api-keys/{api_key_id}", headers=jwt)

    # Disconnected → strict no-op (central facts are an enhancement, never required).
    assert PlaneFactBlock(agent_id=agent_id).render("anything") == []
