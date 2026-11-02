"""End-to-end quality gate — what a framework agent gets from the live plane (1.98.0).

No mocks. A real SDK against the LIVE local plane, doing what
docs/integrations/frameworks.md tells a LangGraph / CrewAI / Pydantic AI user to
do. The framework call itself is not the subject (the SDK imports no framework
code), so the run here is plain code under the recipe's root span. The plane's
own check of these four was ``fastaiagent-enterprise/files_claude/
sdk-notes-frameworks-2026-11-02.md`` (B1, B2, B4, B5):

    push the agent after connect()     -> the enrollment names it: coverage shows
                                          this SDK's version and fail mode on the agent
    a rule attached to that agent      -> plane_guardrails_for_agent(<its name>) returns it
    an approval policy on a tool       -> check_tool_call(...) = require_approval + request id
    prompt.trace_attributes() on root  -> the plane links the span to the prompt version

Setup is the other connected gates' (one persistent lab domain):

    E2E_PLANE_EMAIL / E2E_PLANE_PASSWORD  a domain-admin on the local plane
    FASTAIAGENT_TARGET                    http://localhost:20001
    FASTAIAGENT_API_KEY                   a key in that domain (agent:write, prompt:write)
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.conftest import (
    lab_guardrail_ids,
    plane_admin,
    remove_guardrails_added_since,
    require_env,
    require_platform,
)

pytestmark = pytest.mark.e2e

RUN = uuid.uuid4().hex[:8]
AGENT = f"fw-e2e-{RUN}"
OTHER = f"fw-e2e-other-{RUN}"
TOOL = f"refund_{RUN}"


def _poll(fetch: Any, ok: Any, timeout: float = 30.0) -> Any:
    deadline = time.monotonic() + timeout
    last = fetch()
    while not ok(last) and time.monotonic() < deadline:
        time.sleep(1.0)
        last = fetch()
    return last


def test_a_framework_agent_on_the_plane(isolated_local_db: Path) -> None:
    require_env()
    require_platform()

    import httpx

    import fastaiagent as fa
    from fastaiagent._internal.instance import get_instance_id
    from fastaiagent._version import __version__
    from fastaiagent.prompt import PromptRegistry
    from fastaiagent.trace.otel import get_tracer_provider
    from fastaiagent.trace.platform_export import PlatformSpanExporter

    target = os.environ["FASTAIAGENT_TARGET"].rstrip("/")
    key = os.environ["FASTAIAGENT_API_KEY"]
    admin, jwt, domain_id = plane_admin(
        target,
        purpose="this gate authors a guardrail, an approval policy and a prompt through "
        "the console API, which needs a domain admin.",
    )
    rules_before = lab_guardrail_ids(admin, jwt, domain_id)
    scoped = {"domain_id": domain_id}

    slug = f"fw-e2e-support-{RUN}"
    published = httpx.post(
        f"{target}/public/v1/prompts",
        headers={"X-API-Key": key},
        json={"slug": slug, "content": "You are support for {{company}}.", "category": "agent"},
        timeout=30,
    )
    assert published.status_code in (200, 201), published.text[:200]
    policy = admin.post(
        "/api/v1/approval-policies",
        params=scoped,
        headers=jwt,
        json={
            "name": f"sdk-fw-e2e-{RUN}",
            "description": "SDK framework e2e (safe to delete)",
            "tool_pattern": TOOL,
            "condition_type": "always",
            "timeout_minutes": 1,
        },
    )
    assert policy.status_code == 201, policy.text
    rule = admin.post(
        "/api/v1/guardrails",
        params=scoped,
        headers=jwt,
        json={
            "name": f"fw-e2e-ssn-{RUN}",
            "description": "SDK framework e2e (safe to delete)",
            "guardrail_type": "input",
            "validation_mode": "blocking",
            "implementation_type": "regex",
            "config": {"pattern": r"\b\d{3}-\d{2}-\d{4}\b"},
            "is_active": True,
            "on_error": "block",
        },
    )
    assert rule.status_code < 300, rule.text

    fa.connect(api_key=key, target=target)
    try:
        # ── B5: agents pushed after connect() are on the enrollment ────────────
        agent_id = fa.Agent(name=AGENT, system_prompt="You help.").push().agent_id
        other_id = fa.Agent(name=OTHER, system_prompt="You help.").push().agent_id
        assert agent_id and other_id

        def mine() -> dict[str, Any]:
            cov = admin.get("/api/v1/governance/coverage", headers=jwt, params=scoped)
            assert cov.status_code == 200, cov.text[:200]
            return cov.json()

        def enrolled(cov: dict[str, Any]) -> bool:
            rows = [e for e in cov["enrollments"] if e["instance_id"] == get_instance_id()]
            return bool(rows) and {agent_id, other_id} <= set(rows[0]["governed_agent_ids"] or [])

        coverage = _poll(mine, enrolled)
        assert enrolled(coverage), "the enrollment does not name both pushed agents"
        row = next(a for a in coverage["agents"] if a["agent_id"] == agent_id)
        assert row["sdk_version"] == __version__, row
        assert row["fail_mode"] == "open", row

        # ── B2: a rule attached to the agent, asked for by name ────────────────
        attached = admin.post(
            f"/api/v1/agents/{agent_id}/guardrails",
            headers=jwt,
            json={"guardrail_id": rule.json()["id"]},
        )
        assert attached.status_code < 300, attached.text
        fa.refresh_policy()
        rail_name = rule.json()["name"]
        assert rail_name in [g.name for g in fa.plane_guardrails_for_agent(AGENT)]
        assert rail_name not in [g.name for g in fa.plane_guardrails_for_agent(OTHER)]

        # ── B4: the approval policy, from a tool fastaiagent does not run ──────
        decision = fa.check_tool_call(TOOL, {"order_id": "A100", "amount": 250}, agent=AGENT)
        assert decision.decision == "require_approval", decision
        assert decision.approval_request_id
        assert fa.check_tool_call("lookup_order", {}, agent=AGENT).decision == "allow"

        # ── B1: the run's root span links it to the plane prompt ───────────────
        prompt = PromptRegistry().get(slug, source="platform")
        with fa.trace_context(
            f"agent.{AGENT}",
            **{"agent.name": AGENT, "input.value": "Where is order A100?"},
            **prompt.trace_attributes(),
        ) as root:
            root.set_attribute("output.value", "It shipped yesterday.")
            trace_id = format(root.get_span_context().trace_id, "032x")
        provider = get_tracer_provider()
        if provider is not None:
            provider.force_flush()

        def spans() -> list[dict[str, Any]]:
            PlatformSpanExporter().export([])
            resp = admin.get(f"/api/v1/traces/{trace_id}/spans", headers=jwt, params=scoped)
            return list(resp.json()) if resp.status_code == 200 else []

        served = _poll(spans, lambda s: bool(s))
        refs = [s.get("prompt_ref") for s in served if s.get("prompt_ref")]
        assert refs and refs[0]["prompt_version_id"], f"no span linked to {slug}: {served}"
    finally:
        fa.disconnect()
        remove_guardrails_added_since(admin, jwt, domain_id, rules_before)
        gone = admin.delete(f"/api/v1/approval-policies/{policy.json()['id']}", headers=jwt)
        if gone.status_code == 409:  # it has approval history: deactivate instead
            admin.put(
                f"/api/v1/approval-policies/{policy.json()['id']}",
                headers=jwt,
                json={"is_active": False},
            )
        admin.delete(f"/api/v1/prompts/{slug}", headers=jwt, params={"permanent": "true"})
