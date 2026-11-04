"""Live gate — a Supervisor and a Swarm on a real Enterprise plane (1.99.0).

The plane is real throughout: a prompt from its registry, a hosted KB, an approval
policy, registration, trace ingest. The agents run the SDK's offline
``FunctionModel``, so which tool is called is decided here and not by a model —
the governance check below must not depend on a model choosing to call the tool.

What it holds the SDK to, each found live in the multi-agent check of 1.98.0:

* governance — a member built after ``connect()`` is gated by an
  always-require-approval policy from its second run (it never was again);
* the hosted KB tool returns the whole chunk (it cut each hit to 200 characters);
* a member's model calls carry its registry prompt, and its plane definition keeps
  the slug and leaves out the team's ``handoff_to_*`` tools;
* a streamed Swarm run is ONE trace on the plane (it was one per hop).

Everything it creates on the plane is named per run and removed at the end, apart
from the registered agents (as in the other connected gates).
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from tests.e2e.conftest import (
    lab_guardrail_ids,
    plane_admin,
    remove_guardrails_added_since,
    require_env,
    require_lab_project,
    require_platform,
)

RUN = uuid.uuid4().hex[:6]
TOOL = f"e2e_teams_refund_{RUN}"
FACT = "processing fee of $17.40"
DOC = (
    "This policy applies to every consumer plan sold after January 2026, in every "
    "region and through every sales channel including the app, phone sales and "
    "retail stores, and is maintained by customer operations. "
    f"Annual plans cancelled within 30 days of purchase carry a {FACT}."
)


class _Spans(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[Any] = []

    def export(self, spans):  # type: ignore[override]
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover
        pass


def _poll(fn: Any, ok: Any, timeout: float = 60.0) -> Any:
    deadline, value = time.time() + timeout, None
    while time.time() < deadline:
        value = fn()
        if ok(value):
            return value
        time.sleep(1.0)
    return value


def _refund_route(messages: list[Any]) -> Any:
    if messages[-1].role.value == "tool":
        return f"done: {messages[-1].content}"
    return "", [{"name": TOOL, "arguments": {"order_id": "A100", "amount": 10}}]


def _lead_route(messages: list[Any]) -> Any:
    if messages[-1].role.value == "tool":
        return f"team: {messages[-1].content}"
    return "", [{"name": "delegate_to_refunds", "arguments": {"task": "refund A100"}}]


@pytest.mark.e2e
def test_teams_on_the_plane(tmp_path: Path) -> None:
    require_env()
    require_platform()
    base = os.environ["FASTAIAGENT_TARGET"].rstrip("/")
    key = os.environ["FASTAIAGENT_API_KEY"]
    admin, jwt, domain_id = plane_admin(
        base, purpose="this gate creates a KB and an approval policy, which need a domain admin."
    )
    project_id = require_lab_project(admin, jwt, domain_id)
    scoped = {"domain_id": domain_id}
    rules_before = lab_guardrail_ids(admin, jwt, domain_id)
    slug = f"e2e-teams-{RUN}"
    kb_id = policy_id = None

    import fastaiagent as fa
    from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer
    from fastaiagent.prompt import PromptRegistry
    from fastaiagent.testing import FunctionModel, TestModel
    from fastaiagent.tool import FunctionTool
    from fastaiagent.trace.otel import get_tracer_provider
    from fastaiagent.trace.platform_export import PlatformSpanExporter

    spans = _Spans()
    get_tracer_provider().add_span_processor(SimpleSpanProcessor(spans))
    try:
        published = admin.post(
            "/public/v1/prompts",
            headers={"X-API-Key": key},
            json={"slug": slug, "content": "You issue refunds.", "category": "agent"},
        )
        assert published.status_code in (200, 201), published.text[:200]
        kb = admin.post(
            "/api/v1/knowledge-bases",
            headers=jwt,
            json={"project_id": project_id, "name": slug, "description": "SDK teams e2e"},
        )
        assert kb.status_code == 201, kb.text[:200]
        kb_id = kb.json()["id"]
        doc = admin.post(
            f"/api/v1/knowledge-bases/{kb_id}/documents",
            headers=jwt,
            json={"name": "policy", "source_type": "text", "text_content": DOC},
        )
        assert doc.status_code == 200, doc.text[:200]
        docs = _poll(
            lambda: admin.get(f"/api/v1/knowledge-bases/{kb_id}/documents", headers=jwt).json(),
            lambda ds: ds and ds[0]["status"] in ("completed", "failed"),
        )
        assert docs[0]["status"] == "completed", docs
        policy = admin.post(
            "/api/v1/approval-policies",
            params=scoped,
            headers=jwt,
            json={
                "name": f"sdk-teams-e2e-{RUN}",
                "description": "SDK teams e2e (safe to delete)",
                "tool_pattern": TOOL,
                "condition_type": "always",
                "timeout_minutes": 1,
            },
        )
        assert policy.status_code == 201, policy.text[:200]
        policy_id = policy.json()["id"]

        fa.connect(api_key=key, target=base)
        try:
            # ── the hosted KB tool hands over the whole chunk ─────────────────
            assert DOC.index(FACT) > 200
            out = fa.PlatformKB(kb_id=kb_id).as_tool().execute({"query": "annual refund fee"})
            assert FACT in out.output, out.output

            # ── members built AFTER connect(), with a plane prompt ────────────
            prompt = PromptRegistry().get(slug, source="platform")
            executed: list[str] = []

            def refund(order_id: str, amount: float) -> str:
                executed.append(order_id)
                return f"refunded {amount} on {order_id}"

            def refunder(name: str) -> fa.Agent:
                return fa.Agent(
                    name=f"e2e-teams-{RUN}-{name}",
                    system_prompt=prompt,
                    llm=FunctionModel(_refund_route, model="refunder-model"),
                    tools=[FunctionTool(name=TOOL, fn=refund)],
                )

            sup_member, swarm_member = refunder("worker"), refunder("member")
            lead = fa.Supervisor(
                name=f"e2e-teams-{RUN}-lead",
                llm=FunctionModel(_lead_route),
                workers=[fa.Worker(sup_member, role="refunds")],
                checkpointer=SQLiteCheckpointer(str(tmp_path / "sup.db")),
            )
            entry = fa.Agent(
                name=f"e2e-teams-{RUN}-entry",
                llm=FunctionModel(
                    lambda m: (
                        "",
                        [{"name": f"handoff_to_{swarm_member.name}", "arguments": {"reason": "x"}}],
                    )
                ),
            )
            swarm = fa.Swarm(
                name=f"e2e-teams-{RUN}-swarm",
                agents=[entry, swarm_member],
                entrypoint=entry.name,
                checkpointer=SQLiteCheckpointer(str(tmp_path / "swarm.db")),
            )

            for team, member in ((lead, sup_member), (swarm, swarm_member)):
                team.run("Refund order A100.")  # registers the member on its way
                assert _poll(lambda m=member: m.agent_id, bool, timeout=20), (
                    f"{member.name}: the plane's agent_id never reached the agent"
                )
                executed.clear()
                second = team.run("Refund order A100.")
                assert second.status == "paused", (team, second.status, second.output)
                assert executed == [], f"{member.name}: the gated tool ran unapproved"

            # ── a member's calls carry its prompt; its definition is its own ──
            stamped = [
                s.attributes.get("fastaiagent.prompt.slug")
                for s in spans.spans
                if s.name == "llm.test.refunder-model"
            ]
            assert stamped and set(stamped) == {slug}, stamped
            for member in (sup_member, swarm_member):
                served = admin.get(f"/api/v1/agents/{member.agent_id}", headers=jwt).json()
                assert served["prompt_slug"] == slug, served
                tools = [
                    admin.get(f"/api/v1/tools/{tid}", headers=jwt).json()["name"]
                    for tid in served.get("tool_ids") or []
                ]
                assert tools == [TOOL], f"{member.name} registered with {tools}"

            # ── a streamed swarm run is one trace on the plane ────────────────
            spans.spans.clear()
            s_member = fa.Agent(name=f"e2e-teams-{RUN}-s-member", llm=TestModel("hi"))
            s_entry = fa.Agent(
                name=f"e2e-teams-{RUN}-s-entry",
                llm=FunctionModel(
                    lambda m: (
                        "",
                        [{"name": f"handoff_to_{s_member.name}", "arguments": {"reason": "x"}}],
                    )
                ),
            )
            plain = fa.Swarm(
                name=f"e2e-teams-{RUN}-stream", agents=[s_entry, s_member], entrypoint=s_entry.name
            )

            async def consume() -> None:
                async for _ in plain.astream("hi"):
                    pass

            asyncio.run(consume())
            roots = [s for s in spans.spans if s.parent is None]
            assert [s.name for s in roots] == [f"swarm.{plain.name}"]
            trace_id = format(roots[0].context.trace_id, "032x")
            get_tracer_provider().force_flush()

            def served_spans() -> list[str]:
                PlatformSpanExporter().export([])
                resp = admin.get(f"/api/v1/traces/{trace_id}/spans", headers=jwt, params=scoped)
                return [s["name"] for s in resp.json()] if resp.status_code == 200 else []

            want = {f"agent.{s_entry.name}", f"agent.{s_member.name}"}
            names = _poll(served_spans, lambda n: want <= set(n))
            assert want <= set(names), names
        finally:
            fa.disconnect()
    finally:
        if policy_id:
            gone = admin.delete(f"/api/v1/approval-policies/{policy_id}", headers=jwt)
            if gone.status_code == 409:  # it has approval history: deactivate instead
                admin.put(
                    f"/api/v1/approval-policies/{policy_id}", headers=jwt, json={"is_active": False}
                )
        if kb_id:
            admin.delete(f"/api/v1/knowledge-bases/{kb_id}", headers=jwt)
        admin.delete(f"/api/v1/prompts/{slug}", headers=jwt, params={"permanent": "true"})
        remove_guardrails_added_since(admin, jwt, domain_id, rules_before)
