"""Unit tests for the connected-agent visibility work (no mocks).

Covers the SDK-side halves that need no network:
* Gap 1 — guardrail results emitted as child spans (pass *and* block).
* Gap 4 — registry-prompt provenance retained + stamped on the llm span.
* run(metadata=...) — MLflow-style trace tags on the root span.
* Agent.to_dict() regression + prompt-slug provenance auto-link.
* push helpers — console-URL builder, PushResult, registry tracking.
* trace buffer — prune_acked deletes only acked rows.

Real OpenTelemetry spans through the real provider; the LLM is driven by the
built-in recorded-replay path (no network). The plane round-trip lives in the
env-gated e2e suite (tests/e2e), not here.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

import fastaiagent as fa
from fastaiagent.guardrail import GuardrailPosition, no_pii
from fastaiagent.guardrail.executor import execute_guardrails
from fastaiagent.llm.client import LLMResponse, _replay_recorded_response
from fastaiagent.prompt.prompt import Prompt
from fastaiagent.trace.otel import get_tracer_provider


class _Collector(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[tuple[str, dict, str]] = []

    def export(self, spans):  # type: ignore[override]
        for s in spans:
            self.spans.append((s.name, dict(s.attributes), s.status.status_code.name))
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover
        pass


@pytest.fixture()
def collector() -> _Collector:
    """Attach a real in-memory span collector to the SDK's tracer provider."""
    col = _Collector()
    get_tracer_provider().add_span_processor(SimpleSpanProcessor(col))
    return col


# --------------------------------------------------------------------------- #
# Gap 1 — guardrail spans
# --------------------------------------------------------------------------- #
def test_guardrail_span_on_pass_and_block(collector: _Collector) -> None:
    async def _run() -> None:
        rail = [no_pii(position=GuardrailPosition.output)]
        await execute_guardrails(rail, "all clean here", GuardrailPosition.output)
        with pytest.raises(Exception):
            await execute_guardrails(rail, "My SSN is 123-45-6789.", GuardrailPosition.output)

    asyncio.run(_run())
    guard = [(a, st) for n, a, st in collector.spans if n.startswith("guardrail.")]
    assert len(guard) == 2, "one span per guardrail run (pass + block)"
    by_result = {
        json.loads(a["fastaiagent.guardrail.checks"])[0]["result"]: (a, st) for a, st in guard
    }
    # pass span
    pa, pst = by_result["pass"]
    assert pa["span_type"] == "guardrail"
    assert pa["fastaiagent.guardrail.passed"] is True
    assert pa["fastaiagent.guardrail.position"] == "output"
    assert pst == "OK"
    # block span
    ba, bst = by_result["block"]
    assert ba["fastaiagent.guardrail.passed"] is False
    assert bst == "ERROR"


# --------------------------------------------------------------------------- #
# Gap 4 — prompt provenance
# --------------------------------------------------------------------------- #
def test_prompt_provenance_stamped_on_llm_span(collector: _Collector) -> None:
    p = Prompt(
        name="acme", template="You are {{role}}.", version=7,
        slug="acme-support-system", source="platform", environment="production",
    )
    agent = fa.Agent(
        name="P4Unit", system_prompt=p,
        llm=fa.LLMClient(provider="openai", model="gpt-4o-mini"),
    )
    # Gap 3 synergy: the slug is auto-linked and text is used at runtime.
    assert agent.prompt_slug == "acme-support-system"
    assert agent.system_prompt == "You are {{role}}."

    _replay_recorded_response.set(
        [LLMResponse(content="hi", usage={"prompt_tokens": 1, "completion_tokens": 1},
                     finish_reason="stop", tool_calls=[])]
    )
    agent.run("hello")
    llm = [a for n, a, _ in collector.spans if n.startswith("llm.")]
    assert llm, "an llm span was produced"
    assert llm[0]["fastaiagent.prompt.slug"] == "acme-support-system"
    assert llm[0]["fastaiagent.prompt.version"] == 7
    assert llm[0]["fastaiagent.prompt.environment"] == "production"


def test_no_provenance_when_plain_prompt(collector: _Collector) -> None:
    agent = fa.Agent(name="Plain4", system_prompt="static",
                     llm=fa.LLMClient(provider="openai", model="gpt-4o-mini"))
    _replay_recorded_response.set(
        [LLMResponse(content="hi", usage={"prompt_tokens": 1, "completion_tokens": 1},
                     finish_reason="stop", tool_calls=[])]
    )
    agent.run("hello")
    llm = [a for n, a, _ in collector.spans if n.startswith("llm.")]
    assert llm and "fastaiagent.prompt.slug" not in llm[0]


# --------------------------------------------------------------------------- #
# metadata tags
# --------------------------------------------------------------------------- #
def test_run_metadata_tags_root_span(collector: _Collector) -> None:
    agent = fa.Agent(name="MetaUnit", system_prompt="hi",
                     llm=fa.LLMClient(provider="openai", model="gpt-4o-mini"))
    _replay_recorded_response.set(
        [LLMResponse(content="ok", usage={"prompt_tokens": 1, "completion_tokens": 1},
                     finish_reason="stop", tool_calls=[])]
    )
    agent.run("hello", metadata={"customer": "acme", "n": 3, "tags": ["a", "b"]})
    root = [a for n, a, _ in collector.spans if n.startswith("agent.")][0]
    assert root["fastaiagent.meta.customer"] == "acme"
    assert root["fastaiagent.meta.n"] == 3
    assert json.loads(root["fastaiagent.meta.tags"]) == ["a", "b"]
    assert "user.id" not in root  # the dropped chip must not reappear


# --------------------------------------------------------------------------- #
# Gap 3 — to_dict regression + provenance link
# --------------------------------------------------------------------------- #
def test_to_dict_regression_and_slug_link() -> None:
    plain = fa.Agent(name="Plain", system_prompt="hi").to_dict()
    assert set(plain) == {
        "name", "agent_type", "system_prompt", "llm_endpoint", "tools", "guardrails", "config",
    }
    p = Prompt(name="s", template="t", version=2, slug="my-slug", source="platform")
    d = fa.Agent(name="Slugged", system_prompt=p).to_dict()
    assert d["prompt_slug"] == "my-slug"
    assert d["system_prompt"] == ""


# --------------------------------------------------------------------------- #
# push helpers
# --------------------------------------------------------------------------- #
def test_console_url_and_result_and_tracking() -> None:
    from fastaiagent._platform import push as push_mod
    from fastaiagent.client import _connection

    push_mod.reset_registration_state()
    r = push_mod.PushResult(agent_id="a1", name="X", version=3, url="u")
    assert (r.agent_id, r.name, r.version) == ("a1", "X", 3)

    _connection.console_url = None
    _connection.target = "https://plane.example.net"
    assert push_mod._console_url_for_agent("abc") == "https://plane.example.net/next/agents/abc"
    _connection.console_url = "http://localhost:20000"
    assert push_mod._console_url_for_agent("abc") == "http://localhost:20000/next/agents/abc"
    _connection.console_url = None

    agent = fa.Agent(name="Tracked", system_prompt="hi")
    assert any(a is agent for a in push_mod._agent_registry)


# --------------------------------------------------------------------------- #
# buffer prune
# --------------------------------------------------------------------------- #
def test_prune_acked_deletes_only_synced(tmp_path) -> None:
    from fastaiagent.trace.storage import TraceStore

    db = str(tmp_path / "local.db")
    store = TraceStore(db_path=db)
    conn = store._db
    # Insert 2 acked + 1 unacked span.
    for sid, synced in [("s1", 1), ("s2", 1), ("s3", 0)]:
        conn.execute(
            "INSERT INTO spans (span_id, trace_id, name, start_time, synced) VALUES (?,?,?,?,?)",
            (sid, "t1", "x", "2026-01-01T00:00:00+00:00", synced),
        )
    deleted = store.prune_acked()
    assert deleted == 2
    remaining = conn.fetchall("SELECT span_id, synced FROM spans")
    assert [r["span_id"] for r in remaining] == ["s3"]
    store.close()
