"""End-to-end quality gate — connected governance enrollment (WS4).

No mocks. A real SDK connects to the LIVE local plane and asserts the governance
enrollment round-trip:

    connect()                       -> kicks a fire-and-forget enroll push
    POST /public/v1/governance/enroll -> {enrolled:true, instance_id, fail_mode,
                                          first_seen_at, last_seen_at}
    re-POST (same instance_id)        -> UPSERT: same first_seen_at, refreshed
                                          (>=) last_seen_at  [idempotency proof]

The SDK attests its posture via ``fail_mode``; this gate connects with
``governance_fail_mode="closed"`` so the attested value is observable on the
response (and, optionally, in the console coverage view).

Gated by the ``connected_state_plane`` bundle flag: a 403 on the enroll probe =>
a clean skip with a setup message instead of an opaque failure.

Optional admin-JWT sub-check: when ``E2E_PLANE_EMAIL`` / ``E2E_PLANE_PASSWORD``
are set (a domain-admin on the local plane), it also asserts the enrolled
instance shows up in ``GET /api/v1/governance/coverage`` with the attested
fail_mode. Absent those creds (or if the route shape differs) the sub-check is
skipped, never failed.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from tests.e2e.conftest import require_env, require_platform

pytestmark = pytest.mark.e2e


def test_connected_governance_enroll_roundtrip_and_upsert(isolated_local_db: Any) -> None:
    require_env()
    require_platform()

    import httpx

    import fastaiagent as fa
    from fastaiagent import governance
    from fastaiagent._internal.instance import get_instance_id
    from fastaiagent.client import _connection

    # 1) Connect to the LIVE plane with a real key + target, opting into
    #    fail-closed so the attested fail_mode is observable end to end.
    fa.connect(
        api_key=os.environ["FASTAIAGENT_API_KEY"],
        target=os.environ["FASTAIAGENT_TARGET"],
        governance_fail_mode="closed",
    )
    try:
        assert _connection.is_connected, "connect() did not establish a connection"
        assert _connection.governance_fail_mode == "closed"

        # 2) Feature-gate pre-check — probe enroll, skip cleanly on 403 (mirror the
        #    HITL/durability/memory gates) when connected_state_plane is off.
        instance_id = get_instance_id()
        probe = httpx.post(
            f"{_connection.target}/public/v1/governance/enroll",
            headers=_connection.headers,
            json={"instance_id": instance_id, "fail_mode": "closed"},
            timeout=10,
        )
        if probe.status_code == 403:
            pytest.skip(
                "connected_state_plane is not enabled for this domain — enable the "
                "Enterprise bundle flag on the plane to run this gate "
                f"(probe HTTP {probe.status_code}: {probe.text[:120]})."
            )
        assert probe.status_code < 400, (
            f"enroll probe failed: HTTP {probe.status_code} — {probe.text[:200]}"
        )
        first = probe.json()
        assert first.get("enrolled") is True, first
        assert first.get("instance_id") == instance_id, first
        assert first.get("fail_mode") == "closed", first
        first_seen = first.get("first_seen_at")
        last_seen_1 = first.get("last_seen_at")
        assert first_seen and last_seen_1, first

        # 3) Re-POST via the real SDK enroll() — exercises the production code path
        #    (body composition, 4xx handling, JSON parse) and proves the upsert.
        second = governance.enroll()
        assert second is not None, "enroll() returned None against a live plane"
        assert second.get("enrolled") is True, second
        assert second.get("instance_id") == instance_id, second
        assert second.get("fail_mode") == "closed", second
        # Upsert: first_seen_at preserved, last_seen_at refreshed (monotonic).
        assert second.get("first_seen_at") == first_seen, (first, second)
        assert second.get("last_seen_at") >= last_seen_1, (first, second)

        # 4) OPTIONAL admin-JWT coverage check — skip the SUB-CHECK (not the test)
        #    when creds are absent or the route shape differs.
        email = os.environ.get("E2E_PLANE_EMAIL")
        password = os.environ.get("E2E_PLANE_PASSWORD")
        if email and password:
            base = _connection.target.rstrip("/")
            http = httpx.Client(timeout=30)
            tok = http.post(
                f"{base}/api/v1/auth/login",
                json={"email": email, "password": password},
            ).json()["access_token"]
            jwt = {"Authorization": f"Bearer {tok}"}
            cov = http.get(
                f"{base}/api/v1/governance/coverage",
                headers=jwt,
                params={"domain_id": _connection.domain_id},
            )
            if cov.status_code == 200:
                payload = cov.json()
                enrollments = (
                    payload.get("enrollments")
                    if isinstance(payload, dict)
                    else payload
                ) or []
                mine = [e for e in enrollments if e.get("instance_id") == instance_id]
                assert mine, f"instance {instance_id} not present in coverage: {payload}"
                assert mine[0].get("fail_mode") == "closed", mine[0]
            # else: endpoint not exposed / shape differs — don't fail the gate.
    finally:
        fa.disconnect()
