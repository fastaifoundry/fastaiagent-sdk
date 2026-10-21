"""No credential survives serialization — the S1 sweep (1.89.0).

The defect. ``config["llm"]`` on a model-backed guardrail (and ``decision["llm"]``
on a Chain condition node) is ``LLMClient`` kwargs, stored as a plain dict so the
rule can be distributed. A key put there was copied verbatim by ``to_dict()``
into the root span's ``agent.guardrails`` attribute — which is not a payload key,
so it reached ``local.db``, the plane and every OTLP exporter even with
``FASTAIAGENT_TRACE_PAYLOADS=0`` — and into the agent definition pushed to the
plane, which stored it.

The design (decided 2026-10-14). A shared rule never carries a credential; the
process that *runs* the rule supplies its own. So:

1. **Construction refuses** an ``llm`` dict holding anything outside the
   allowlist of shareable ``LLMClient`` fields, and the error never echoes the
   value.
2. **Rebuild drops** — a rule read back from a trace or the plane (stored by an
   older SDK) still runs, without the key, and falls back to the env var.
3. **Serialization applies the allowlist again**, so a config mutated after
   construction still cannot carry a key out.

Section 3 is the sweep that closes the *class*: every place a credential can
be handed to the SDK, run under tracing, then every channel that leaves the
process is searched for the secret.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from typing import Any

import pytest

from fastaiagent.guardrail import Guardrail, GuardrailType
from fastaiagent.guardrail.guardrail import GuardrailPosition

#: Never a real key. Long and distinctive so a substring hit is unambiguous.
SECRET = "sk-sweep-0123456789abcdefSECRETdoNotLeak"

#: Credential-bearing names a caller might reasonably try in an ``llm`` dict.
CREDENTIAL_KEYS = ["api_key", "default_headers", "authorization", "aws_secret_access_key"]


def _llm(key: str, **extra: Any) -> dict[str, Any]:
    return {"model": "gpt-4o-mini", key: SECRET, **extra}


# ---------------------------------------------------------------------------
# 1. Construction refuses a credential in a shareable config
# ---------------------------------------------------------------------------


def _banned_topics(llm: dict[str, Any]) -> Any:
    from fastaiagent.guardrail.builtins import banned_topics

    return banned_topics(["weapons"], llm=llm)


def _allowed_topics(llm: dict[str, Any]) -> Any:
    from fastaiagent.guardrail.builtins import allowed_topics

    return allowed_topics(["billing"], llm=llm)


def _llm_judge(llm: dict[str, Any]) -> Any:
    return Guardrail(
        name="judge",
        guardrail_type=GuardrailType.llm_judge,
        config={"prompt": "Is this safe?", "llm": llm},
    )


def _content_safety(llm: dict[str, Any]) -> Any:
    return Guardrail(
        name="cs",
        guardrail_type=GuardrailType.content_safety,
        config={"categories": ["S1"], "llm": llm},
    )


def _groundedness(llm: dict[str, Any]) -> Any:
    return Guardrail(
        name="gr",
        guardrail_type=GuardrailType.groundedness,
        config={"llm": llm},
    )


def _chain_decision(llm: dict[str, Any]) -> Any:
    from fastaiagent.chain import Chain, NodeType
    from fastaiagent.llm.decisions import Choice

    chain = Chain("routing")
    chain.add_node(
        "triage",
        type=NodeType.condition,
        decision={"question": Choice(instructions="Which team?", options=["a", "b"]), "llm": llm},
    )
    return chain


BUILDERS = {
    "banned_topics": _banned_topics,
    "allowed_topics": _allowed_topics,
    "llm_judge": _llm_judge,
    "content_safety": _content_safety,
    "groundedness": _groundedness,
    "chain_decision": _chain_decision,
}


class TestConstructionRefuses:
    @pytest.mark.parametrize("key", CREDENTIAL_KEYS)
    @pytest.mark.parametrize("builder", list(BUILDERS), ids=list(BUILDERS))
    def test_a_credential_in_the_llm_dict_is_refused(self, builder: str, key: str) -> None:
        with pytest.raises(ValueError) as exc:
            BUILDERS[builder](_llm(key))
        message = str(exc.value)
        assert key in message, "the error must name the field it refused"
        assert SECRET not in message, "the refusal must never echo the credential"

    @pytest.mark.parametrize("builder", list(BUILDERS), ids=list(BUILDERS))
    def test_shareable_fields_are_accepted(self, builder: str) -> None:
        llm = {"model": "gpt-4o-mini", "temperature": 0.0, "base_url": "http://127.0.0.1:9/v1"}
        built = BUILDERS[builder](llm)
        assert built is not None

    def test_a_live_client_keeps_the_rule_local_and_is_allowed(self) -> None:
        """The documented way to use a key on a model-backed rule."""
        from fastaiagent.guardrail.builtins import banned_topics
        from fastaiagent.llm import LLMClient

        rule = banned_topics(["weapons"], llm=LLMClient(model="gpt-4o-mini", api_key=SECRET))
        assert SECRET not in json.dumps(rule.to_dict())

    def test_the_allowlist_covers_everything_llmclient_serializes(self) -> None:
        """The allowlist is the shareable half of ``LLMClient``: every field its own
        ``to_dict()`` emits must be accepted, or a round-trip would be refused."""
        from fastaiagent._internal.scrub import LLM_CONFIG_FIELDS
        from fastaiagent.llm import LLMClient

        client = LLMClient(
            provider="openai",
            model="m",
            base_url="http://127.0.0.1:9/v1",
            temperature=0.1,
            max_tokens=10,
            max_retries=2,
            top_p=0.5,
            stop=["x"],
            seed=1,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            parallel_tool_calls=False,
            pdf_mode="vision",
            max_pdf_pages=3,
            max_image_size_mb=1.0,
            # 1.91.0 keys — missing here, the allowlist missed them too (X22).
            reasoning="low",
            cache=True,
            api="responses",
        )
        assert {"reasoning", "cache", "api"} <= set(client.to_dict())
        assert set(client.to_dict()) <= LLM_CONFIG_FIELDS
        bedrock = LLMClient(provider="bedrock", model="m", region="eu-west-1")
        assert set(bedrock.to_dict()) <= LLM_CONFIG_FIELDS
        assert "api_key" not in LLM_CONFIG_FIELDS


# ---------------------------------------------------------------------------
# 2. A rule rebuilt from stored data drops the key and still runs
# ---------------------------------------------------------------------------


def _stored_topic_rule() -> dict[str, Any]:
    """A topic rule as an SDK older than 1.89.0 serialized it — key included."""
    return {
        "name": "no-weapons",
        "guardrail_type": "topic",
        "position": "input",
        "config": {
            "topics": [{"name": "weapons", "description": ""}],
            "mode": "deny",
            "llm": {"model": "gpt-4o-mini", "api_key": SECRET},
        },
    }


class TestRebuildDrops:
    def test_from_dict(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            rule = Guardrail.from_dict(_stored_topic_rule())
        assert rule.config["llm"] == {"model": "gpt-4o-mini"}
        assert "api_key" in caplog.text, "dropping a field must be said out loud"
        assert SECRET not in caplog.text

    def test_from_policy(self, caplog: pytest.LogCaptureFixture) -> None:
        from fastaiagent.guardrail.from_policy import guardrail_from_policy_rule

        stored = _stored_topic_rule()
        rule_payload = {
            "name": stored["name"],
            "implementation_type": "topic",
            "guardrail_type": "input",
            "config": stored["config"],
        }
        with caplog.at_level(logging.WARNING):
            rule = guardrail_from_policy_rule(rule_payload)
        assert rule is not None
        assert rule.config["llm"] == {"model": "gpt-4o-mini"}
        assert SECRET not in caplog.text

    def test_chain_from_dict(self) -> None:
        from fastaiagent.chain import Chain

        stored = {
            "name": "routing",
            "nodes": [
                {
                    "id": "triage",
                    "type": "condition",
                    "config": {
                        "decision": {
                            "question": {"type": "choice", "instructions": "Which team?"},
                            "input": "{{input}}",
                            "llm": {"model": "gpt-6-luna", "api_key": SECRET},
                            "min_confidence": 0.0,
                        }
                    },
                }
            ],
            "edges": [],
        }
        chain = Chain.from_dict(stored)
        assert chain.nodes[0].config["decision"]["llm"] == {"model": "gpt-6-luna"}

    def test_the_rebuilt_rule_authenticates_with_the_env_key(
        self, decisions_stub: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Behaviour, not shape: the request the rebuilt rule sends carries this
        process's key, never the one an older SDK stored in the config."""
        from fastaiagent.guardrail.implementations import run_guardrail

        monkeypatch.setenv("OPENAI_API_KEY", "env-key-of-the-running-process")
        stored = {
            "name": "cs",
            "guardrail_type": "content_safety",
            "position": "output",
            "config": {
                "backend": "decisions",
                "categories": ["S1"],
                "llm": {
                    "provider": "custom",
                    "model": "gpt-6-luna",
                    "base_url": decisions_stub.base_url,
                    "api_key": SECRET,
                },
            },
        }
        rule = Guardrail.from_dict(stored)
        asyncio.run(run_guardrail(rule, "some text"))

        assert decisions_stub.requests, "the rebuilt rule never called its judge"
        assert decisions_stub.requests[-1]["auth"] == "Bearer env-key-of-the-running-process"


# ---------------------------------------------------------------------------
# 3. The sweep: every carrier, every channel out
# ---------------------------------------------------------------------------


def _carriers() -> dict[str, Any]:
    """Every place a caller can hand the SDK a credential, built legally.

    The two shareable configs are mutated *after* construction: construction now
    refuses a key, so this is the remaining way one can sit in ``config`` — and
    serialization must still not carry it out.
    """
    from fastaiagent.agent.memory import ComposableMemory
    from fastaiagent.agent.memory_blocks import FactExtractionBlock
    from fastaiagent.guardrail.builtins import banned_topics
    from fastaiagent.llm import LLMClient
    from fastaiagent.tool.mcp import MCPTool
    from fastaiagent.tool.rest import RESTTool

    topic = banned_topics(["weapons"], llm={"model": "gpt-4o-mini"})
    topic.position = GuardrailPosition.tool_call  # never fires: the test model calls no tools
    topic.config["llm"]["api_key"] = SECRET
    judge = Guardrail(
        name="judge",
        guardrail_type=GuardrailType.llm_judge,
        position=GuardrailPosition.tool_result,
        config={"prompt": "Is this safe?", "llm": {"model": "gpt-4o-mini"}},
    )
    judge.config["llm"]["default_headers"] = {"Authorization": f"Bearer {SECRET}"}
    return {
        "guardrails": [topic, judge],
        "tools": [
            RESTTool(
                name="lookup",
                url="http://127.0.0.1:9/lookup",
                headers={"Authorization": f"Bearer {SECRET}"},
            ),
            MCPTool(
                name="remote",
                server_url="http://127.0.0.1:9/mcp",
                tool_name="t",
                auth_token=SECRET,
            ),
        ],
        "memory": ComposableMemory(
            blocks=[FactExtractionBlock(llm=LLMClient(model="gpt-4o-mini", api_key=SECRET))]
        ),
    }


def _agent(name: str, *, with_memory: bool = False) -> Any:
    import fastaiagent as fa
    from fastaiagent.testing import TestModel

    c = _carriers()
    model = TestModel()
    model.api_key = SECRET  # an agent LLM configured with an explicit key
    return fa.Agent(
        name=name,
        system_prompt="You help.",
        llm=model,
        tools=c["tools"],
        guardrails=c["guardrails"],
        memory=c["memory"] if with_memory else None,
    )


def _every_row(db_path: Any) -> str:
    """Every value in every table of the local store, as one string."""
    conn = sqlite3.connect(str(db_path))
    try:
        out = []
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for table in tables:
            for row in conn.execute(f'SELECT * FROM "{table}"'):  # noqa: S608 — table names from sqlite_master
                out.append(repr(row))
        return "\n".join(out)
    finally:
        conn.close()


@pytest.fixture
def traced_db(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A private local.db with a tracer provider pinned to it."""
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    monkeypatch.delenv("FASTAIAGENT_API_KEY", raising=False)
    monkeypatch.delenv("FASTAIAGENT_TARGET", raising=False)
    reset_config()
    _project.set_project_id("test-proj")
    otel.reset()
    try:
        yield db_path
    finally:
        otel.reset()
        _project.reset_for_testing()
        reset_config()


class TestSweep:
    def test_definitions_carry_no_secret(self) -> None:
        from fastaiagent.agent.swarm import Swarm
        from fastaiagent.agent.team import Supervisor, Worker
        from fastaiagent.testing import TestModel

        a = _agent("sweep-def", with_memory=True)
        b = _agent("sweep-def-b")
        swarm = Swarm(name="sw", agents=[a, b], entrypoint=a.name)
        sup = Supervisor(name="sup", llm=TestModel(), workers=[Worker(agent=a)])
        chain = _chain_decision({"model": "gpt-6-luna"})
        chain.nodes[0].config["decision"]["llm"]["api_key"] = SECRET

        for label, payload in {
            "Agent.to_dict": a.to_dict(),
            "Swarm.to_dict": swarm.to_dict(),
            "Supervisor.to_dict": sup.to_dict(),
            "Chain.to_dict": chain.to_dict(),
        }.items():
            assert SECRET not in json.dumps(payload, default=str), f"{label} carries the secret"

    def test_a_traced_run_leaks_through_no_channel(
        self, traced_db: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("opentelemetry.sdk.trace")
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from fastaiagent.trace import otel
        from fastaiagent.trace.platform_export import to_wire
        from fastaiagent.trace.storage import TraceStore

        third_party = InMemorySpanExporter()
        otel.add_span_processor(SimpleSpanProcessor(otel._EgressFilteredExporter(third_party)))

        agent = _agent("sweep-run")
        result = agent.run("hello")
        assert result.trace_id, "the run was not traced, so the sweep would check nothing"

        spans = TraceStore.default().get_trace(result.trace_id).spans
        root = next(s for s in spans if s.parent_span_id is None)
        assert "agent.guardrails" in root.attributes, "the carrier attribute was not recorded"

        # local.db — every table, every row.
        assert SECRET not in _every_row(traced_db), "local.db holds the secret"

        # The plane wire, with payloads exported and with them gated.
        for payloads in ("1", "0"):
            monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", payloads)
            wire = json.dumps([to_wire(s) for s in spans], default=str)
            assert SECRET not in wire, f"to_wire carries the secret (payloads={payloads})"

        # A third-party OTLP exporter behind add_exporter's egress filter.
        exported = json.dumps(
            [dict(s.attributes or {}) for s in third_party.get_finished_spans()], default=str
        )
        assert third_party.get_finished_spans(), "the third-party exporter saw nothing"
        assert SECRET not in exported, "an add_exporter sink received the secret"

    def test_the_agent_push_carries_no_secret(self, capture_server: Any) -> None:
        import fastaiagent as fa
        from fastaiagent._platform import push as push_mod

        push_mod.reset_registration_state()
        fa.connect(api_key="fa_k_test", target=capture_server.url, auto_register=False)
        try:
            agent = _agent("sweep-push", with_memory=True)
            push_mod.push_agent(agent, force=True, best_effort=True)
        finally:
            fa.disconnect()
            push_mod.reset_registration_state()

        pushes = [r for r in capture_server.requests if r["path"].endswith("/sdk/agents")]
        assert pushes, "the agent definition was never pushed"
        assert SECRET not in json.dumps(pushes), "the pushed agent definition carries the secret"
