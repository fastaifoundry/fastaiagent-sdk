"""End-to-end gate — connected-agent visibility against a real Enterprise plane.

Exercises all four gaps in one run against a live plane (no mocks):
* Gap 2 — connect() + run() auto-registers the agent (``agent_id`` populated).
* Gap 3 — the governed definition shows the prompt slug (not "Inline") + memory.
* Gap 1 — the emitted trace carries a ``span_type=guardrail`` span with checks.
* Gap 4 — the llm span carries ``fastaiagent.prompt.slug`` / ``version``.
* metadata — ``run(metadata=...)`` lands as ``fastaiagent.meta.*`` on the root.

Reads ``FASTAIAGENT_TARGET`` + ``E2E_PLANE_EMAIL`` / ``E2E_PLANE_PASSWORD`` from
the env and skips cleanly when absent (hard-fails only in CI via E2E_REQUIRED).
"""

from __future__ import annotations

import json
import os
import uuid

import httpx
import pytest

from tests.e2e.conftest import require_env, require_platform


@pytest.mark.e2e
def test_connected_agent_fully_visible() -> None:
    require_env()
    require_platform()

    email = os.environ.get("E2E_PLANE_EMAIL")
    password = os.environ.get("E2E_PLANE_PASSWORD")
    if not (email and password):
        pytest.skip("E2E_PLANE_EMAIL / E2E_PLANE_PASSWORD not set")

    base = os.environ["FASTAIAGENT_TARGET"].rstrip("/")
    console = os.environ.get("FASTAIAGENT_CONSOLE_URL", base)
    http = httpx.Client(timeout=30)

    # --- mint a scoped key ---------------------------------------------------
    tok = http.post(
        f"{base}/api/v1/auth/login", json={"email": email, "password": password}
    ).json()["access_token"]
    jwt = {"Authorization": f"Bearer {tok}"}
    domain_id = http.get(f"{base}/api/v1/users/me/domains", headers=jwt).json()[0]["id"]
    http.post(
        f"{base}/api/v1/domains/{domain_id}/projects",
        headers=jwt,
        json={"name": "Conn Vis E2E", "slug": "conn-vis-e2e"},
    )
    lk = http.get(f"{base}/api/v1/api-keys", headers=jwt)
    if lk.status_code == 200:
        keys = lk.json()
        keys = keys if isinstance(keys, list) else keys.get("keys", [])
        for k in keys:
            if str(k.get("name", "")).startswith("conn-vis-e2e") and k.get("id"):
                http.delete(f"{base}/api/v1/api-keys/{k['id']}", headers=jwt)
    kr = http.post(
        f"{base}/api/v1/api-keys",
        headers=jwt,
        json={
            "name": f"conn-vis-e2e-{uuid.uuid4().hex[:6]}",
            "permissions": ["read", "write", "execute"],
            "domain_id": domain_id,
            "scopes": ["agent:write", "agent:execute", "prompt:write", "prompt:read"],
        },
    )
    assert kr.status_code == 201, f"key mint failed: {kr.status_code} {kr.text[:160]}"
    api_key = kr.json()["key"]
    api_key_id = kr.json().get("id")

    try:
        slug = "acme-support-system"
        http.post(
            f"{base}/public/v1/prompts",
            headers={"X-API-Key": api_key},
            json={"slug": slug, "content": "You are {{role}} for Acme.", "category": "agent"},
        )

        import fastaiagent as fa
        from fastaiagent.agent.memory import AgentMemory
        from fastaiagent.guardrail import GuardrailPosition, no_pii
        from fastaiagent.prompt import PromptRegistry
        from fastaiagent.trace.storage import TraceStore

        fa.connect(api_key=api_key, target=base, console_url=console)
        try:
            prompt = PromptRegistry().get(slug, source="platform")
            assert prompt.slug == slug and prompt.source == "platform"

            agent = fa.Agent(
                name=f"Conn Vis E2E {uuid.uuid4().hex[:4]}",
                system_prompt=prompt,
                llm=fa.LLMClient(provider="openai", model="gpt-4o-mini"),
                guardrails=[no_pii(position=GuardrailPosition.output)],
                memory=AgentMemory(),
            )
            agent.run(
                "Greet the customer in one short sentence.",
                metadata={"customer": "acme", "env": "e2e"},
            )

            # Gap 1 / Gap 4 / metadata — assert on the emitted spans (the wire).
            rows = TraceStore()._db.fetchall(
                "SELECT name, attributes FROM spans ORDER BY start_time DESC LIMIT 40"
            )
            attrs = [json.loads(r["attributes"] or "{}") for r in rows]
            assert any(
                a.get("span_type") == "guardrail" and "fastaiagent.guardrail.checks" in a
                for a in attrs
            ), "Gap 1: no guardrail span with checks emitted"
            assert any(a.get("fastaiagent.prompt.slug") == slug for a in attrs), (
                "Gap 4: no llm span carrying the prompt slug"
            )
            assert any("fastaiagent.meta.customer" in a for a in attrs), (
                "metadata: fastaiagent.meta.* not stamped"
            )
        finally:
            fa.disconnect()

        # Gap 2 — auto-registered.
        assert agent.agent_id, "Gap 2: agent was not auto-registered (agent_id is None)"

        # Gap 3 — governed definition carries the slug + memory.
        gov = http.get(
            f"{base}/api/v1/agents/{agent.agent_id}/governance", headers=jwt
        )
        assert gov.status_code == 200, f"governance HTTP {gov.status_code}: {gov.text[:160]}"
        consumes = gov.json().get("consumes", gov.json())
        assert consumes.get("prompt_slug") == slug, (
            f"Gap 3: prompt_slug={consumes.get('prompt_slug')} (expected {slug})"
        )
        assert consumes.get("memory_enabled") is True, "Gap 3: memory_enabled not True"
    finally:
        if api_key_id:
            http.delete(f"{base}/api/v1/api-keys/{api_key_id}", headers=jwt)
