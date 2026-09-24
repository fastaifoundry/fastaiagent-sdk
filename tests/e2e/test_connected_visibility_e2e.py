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

from tests.e2e.conftest import (
    lab_guardrail_ids,
    plane_admin,
    remove_guardrails_added_since,
    require_env,
    require_platform,
)


@pytest.mark.e2e
def test_connected_agent_fully_visible() -> None:
    require_env()
    require_platform()

    base = os.environ["FASTAIAGENT_TARGET"].rstrip("/")
    console = os.environ.get("FASTAIAGENT_CONSOLE_URL", base)
    http = httpx.Client(timeout=30)

    # --- the lab session and key ----------------------------------------------
    # The lab domain and one shared login (the plane rate-limits logins). This
    # gate used ``domains[0]``, which on the lab account is another domain, and
    # created a project and minted a key per run there until the plan's caps
    # refused them (HTTP 402). It now uses the lab's own key.
    admin, jwt, domain_id = plane_admin(
        base,
        purpose="this gate reads the governed definition through the console API, "
        "which needs a domain admin.",
    )
    api_key = os.environ["FASTAIAGENT_API_KEY"]
    # The agent below carries ``no_pii`` and auto-registers, and the plane installs
    # a pushed agent's guardrails as rules every later run in the domain receives.
    # Leave the lab's rules exactly as found — a leftover ``no_pii`` blocked the
    # whole guardrail-actions gate.
    rules_before = lab_guardrail_ids(admin, jwt, domain_id)

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
        remove_guardrails_added_since(admin, jwt, domain_id, rules_before)
