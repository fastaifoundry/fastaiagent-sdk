"""End-to-end quality gate — OTLP export to a real observability backend.

Proves that the SDK's OTel spans reach an external OTLP collector
end-to-end. Specifically, the round trip is:

    agent.run() → Phase A span instrumentation → OTel TracerProvider
    → BatchSpanProcessor → OTLPSpanExporter (http://localhost:4318)
    → Jaeger ingest → Jaeger storage → Jaeger query API (:16686)
    → this test queries the API and asserts the spans are visible.

Two independent HTTP paths are exercised:
- **Ingest** (push, fastaiagent → Jaeger via OTLP HTTP)
- **Query** (pull, this test → Jaeger API)

Both have to be alive and correct for the gate to pass.

Environment assumption: a Jaeger all-in-one container (or any OTLP
HTTP collector + query backend) is running locally with:
    :4318  → OTLP HTTP receiver  (POST /v1/traces)
    :16686 → Jaeger query API    (GET  /api/services, /api/traces)

``require_otlp_endpoint()`` probes both and skips the gate cleanly if
either is unreachable. GitHub runners do not have Jaeger by default,
so this gate is effectively a local-only check — same shape as the
Ollama gate.

The gate uses a mocked ``LLMClient.acomplete`` so no real OpenAI calls
are made. OTLP export is about trace shipping, not LLM quality — the
assertion surface is the span tree that lands in Jaeger, not the text
output of an agent. The mocked path still fully exercises the agent
loop, Phase A span instrumentation, tool invocation, and trace storage.

Uses a unique ``service.name`` per test run via ``OTEL_SERVICE_NAME``
so repeated runs on the same Jaeger instance are not confused with
each other — querying Jaeger for our specific service finds exactly
this run's spans.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import httpx
import pytest

from tests.e2e.conftest import require_env, require_otlp_endpoint

pytestmark = pytest.mark.e2e


_OTLP_INGEST = "http://localhost:4318/v1/traces"
_JAEGER_QUERY = "http://localhost:16686/api/traces"


def _lookup_order(order_id: str) -> str:
    """A simple tool that will show up as a tool.lookup_order span."""
    return f"Order {order_id}: delivered"


@pytest.fixture(scope="module")
def otlp_run_context():
    """Set up a unique service.name, reset the tracer provider so the
    new service name is picked up, attach the OTLP exporter, yield the
    service name, and tear down on exit.

    Module scope is important: resetting the tracer provider mid-session
    is disruptive to other tests, so we do it once for this module and
    leave the provider in a clean state on teardown.
    """
    require_env()
    require_otlp_endpoint()

    from fastaiagent.llm.client import LLMClient, LLMResponse
    from fastaiagent.trace import add_exporter, reset
    from fastaiagent.trace.export import create_otlp_exporter

    service_name = f"fastaiagent-otlp-gate-{uuid.uuid4().hex[:10]}"

    # 1. Set service name via OTEL_SERVICE_NAME — TracerProvider() picks
    # this up from os.environ via Resource.create() when it's created.
    prev_service = os.environ.get("OTEL_SERVICE_NAME")
    os.environ["OTEL_SERVICE_NAME"] = service_name

    # 2. Reset the global tracer provider so the next get_tracer_provider()
    # call creates a fresh one with the new service name.
    reset()

    # 3. Attach the OTLP exporter. This creates a BatchSpanProcessor
    # wrapping the exporter and adds it to the (newly created) provider.
    exporter = create_otlp_exporter(endpoint=_OTLP_INGEST)
    add_exporter(exporter)

    # 4. Monkeypatch LLMClient._call_openai — the provider-dispatch
    # function that acomplete delegates to — rather than acomplete
    # itself. Patching acomplete would bypass the OTel span wrap
    # added in PR #1, so no llm.* spans would be emitted. Patching
    # the provider function preserves the wrap AND lets us return a
    # realistic tool_calls payload to trigger the agent's tool loop,
    # which in turn emits the tool.* span in the executor.
    #
    # Sequence the agent goes through:
    #   call 1 → fake returns tool_calls=[lookup_order(GATE-1)]
    #         → agent executor invokes the real FunctionTool
    #           (emits tool.lookup_order span)
    #         → tool result appended to messages
    #   call 2 → fake returns final content, tool_calls=[]
    #         → agent loop exits
    # Net spans: 1 agent root + 2 llm + 1 tool = 4.
    from fastaiagent.llm.message import ToolCall

    original_call_openai = LLMClient._call_openai
    call_count = {"n": 0}

    async def fake_call_openai(self, messages, tools=None, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_otlp_1",
                        name="lookup_order",
                        arguments={"order_id": "GATE-1"},
                    )
                ],
                usage={"total_tokens": 11, "prompt_tokens": 7, "completion_tokens": 4},
                model=self.model,
                finish_reason="tool_calls",
                latency_ms=1,
            )
        return LLMResponse(
            content="Order GATE-1 is delivered.",
            tool_calls=[],
            usage={"total_tokens": 22, "prompt_tokens": 14, "completion_tokens": 8},
            model=self.model,
            finish_reason="stop",
            latency_ms=1,
        )

    LLMClient._call_openai = fake_call_openai  # type: ignore[method-assign]

    try:
        yield service_name
    finally:
        LLMClient._call_openai = original_call_openai  # type: ignore[method-assign]
        # Restore env and reset the provider for subsequent test modules.
        if prev_service is None:
            os.environ.pop("OTEL_SERVICE_NAME", None)
        else:
            os.environ["OTEL_SERVICE_NAME"] = prev_service
        reset()


def _flush_spans(timeout_ms: int = 5000) -> None:
    """Force-flush the global TracerProvider so pending spans leave the
    BatchSpanProcessor before the test polls Jaeger."""
    from fastaiagent.trace.otel import get_tracer_provider

    get_tracer_provider().force_flush(timeout_millis=timeout_ms)


def _query_jaeger_traces(
    service: str, operation: str | None = None, max_attempts: int = 10
) -> list[dict[str, Any]]:
    """Poll Jaeger's /api/traces endpoint until it returns traces or
    until max_attempts is exhausted."""
    params: dict[str, Any] = {"service": service, "limit": 20}
    if operation:
        params["operation"] = operation
    last_traces: list[dict[str, Any]] = []
    for _ in range(max_attempts):
        try:
            resp = httpx.get(_JAEGER_QUERY, params=params, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                traces = data.get("data") or []
                if traces:
                    return traces
                last_traces = traces
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    return last_traces


class TestOTLPExportGate:
    """Real OTLP round trip: fastaiagent → Jaeger → query."""

    def test_01_create_otlp_exporter_returns_real_exporter(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        require_otlp_endpoint()
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        from fastaiagent.trace.export import create_otlp_exporter

        exporter = create_otlp_exporter(endpoint=_OTLP_INGEST)
        assert isinstance(exporter, OTLPSpanExporter), (
            f"create_otlp_exporter returned {type(exporter).__name__}, "
            f"expected OTLPSpanExporter"
        )

    def test_02_add_exporter_is_idempotent(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        require_otlp_endpoint()
        from fastaiagent.trace import add_exporter
        from fastaiagent.trace.export import create_otlp_exporter

        # Adding the same exporter twice must not raise. The underlying
        # TracerProvider tolerates multiple processors with the same
        # exporter — they just dual-emit, which is fine.
        add_exporter(create_otlp_exporter(endpoint=_OTLP_INGEST))
        add_exporter(create_otlp_exporter(endpoint=_OTLP_INGEST))

    def test_03_agent_run_lands_in_jaeger(
        self, otlp_run_context: str, gate_state: dict[str, Any]
    ) -> None:
        """The main round-trip assertion.

        Run a real agent (mocked LLM) with a real tool, force-flush
        spans, poll Jaeger query API for our service, and assert the
        expected span tree is visible.
        """
        require_env()
        require_otlp_endpoint()
        from fastaiagent import Agent, FunctionTool, LLMClient

        service_name = otlp_run_context
        # Use a unique agent name so the root span name is distinctive
        # and we can tell this run apart from any others that might be
        # in Jaeger under the same service.
        agent_name = f"otlp-gate-{uuid.uuid4().hex[:8]}"
        agent = Agent(
            name=agent_name,
            system_prompt="You are a gate test agent. Always reply briefly.",
            llm=LLMClient(provider="openai", model="gpt-4.1-mini"),
            tools=[FunctionTool(name="lookup_order", fn=_lookup_order)],
        )
        result = agent.run("Ping the OTLP gate.")
        assert result.trace_id, "agent.run did not produce a trace_id"
        gate_state["otlp_trace_id"] = result.trace_id
        gate_state["otlp_agent_name"] = agent_name
        gate_state["otlp_service_name"] = service_name

        # Flush pending spans out of the BatchSpanProcessor to Jaeger.
        _flush_spans(timeout_ms=5000)
        # Jaeger ingest → storage → query has a small async gap; give
        # it a moment before the first poll.
        time.sleep(0.5)

        traces = _query_jaeger_traces(
            service=service_name, operation=f"agent.{agent_name}"
        )
        assert traces, (
            f"No traces found in Jaeger for service={service_name!r} "
            f"operation=agent.{agent_name} — the OTLP round-trip broke "
            f"somewhere between the BatchSpanProcessor and the Jaeger "
            f"query API"
        )
        gate_state["jaeger_traces"] = traces

    def test_04_jaeger_trace_carries_phase_a_attributes(
        self, gate_state: dict[str, Any]
    ) -> None:
        """The agent root span in Jaeger must carry the Phase A
        reconstruction attributes (agent.name, agent.input, agent.output,
        agent.config, agent.tools, agent.llm.config). This proves that
        the enriched spans from PR #1 survive the OTLP wire format."""
        require_env()
        require_otlp_endpoint()

        traces = gate_state.get("jaeger_traces")
        assert traces, "test_03 did not populate gate_state['jaeger_traces']"
        agent_name = gate_state["otlp_agent_name"]

        # Find the span whose operationName is agent.<our agent name>.
        agent_span = None
        for trace in traces:
            for span in trace.get("spans", []):
                if span.get("operationName") == f"agent.{agent_name}":
                    agent_span = span
                    break
            if agent_span is not None:
                break
        assert agent_span is not None, (
            f"no span with operationName=agent.{agent_name} found in "
            f"Jaeger traces: {[s.get('operationName') for trace in traces for s in trace.get('spans', [])]}"
        )

        # Jaeger stores span attributes as a list of {key, type, value}
        # dicts under "tags". Flatten them into a dict for easier
        # assertions.
        tags = {t["key"]: t.get("value") for t in agent_span.get("tags", [])}
        for required in (
            "agent.name",
            "agent.input",
            "agent.output",
            "agent.config",
            "agent.tools",
            "agent.llm.config",
            "agent.llm.provider",
            "agent.llm.model",
        ):
            assert required in tags, (
                f"Phase A attribute {required!r} missing from Jaeger span; "
                f"tags: {sorted(tags.keys())}"
            )
        assert tags["agent.name"] == agent_name

    def test_05_tool_and_llm_spans_visible_in_jaeger(
        self, gate_state: dict[str, Any]
    ) -> None:
        """The full span tree (root agent + child tool + child llm) must
        propagate through OTLP. Proves children are not being orphaned
        or dropped by the exporter."""
        require_env()
        require_otlp_endpoint()

        traces = gate_state.get("jaeger_traces")
        assert traces, "test_03 did not populate gate_state['jaeger_traces']"

        all_operations: set[str] = set()
        for trace in traces:
            for span in trace.get("spans", []):
                op = span.get("operationName")
                if op:
                    all_operations.add(op)

        assert any(op.startswith("tool.") for op in all_operations), (
            f"no tool.* span in Jaeger traces — tool instrumentation "
            f"did not propagate through OTLP export. Operations seen: "
            f"{sorted(all_operations)}"
        )
        assert any(op.startswith("llm.") for op in all_operations), (
            f"no llm.* span in Jaeger traces — LLMClient.acomplete span "
            f"wrap did not propagate through OTLP export. Operations "
            f"seen: {sorted(all_operations)}"
        )
        assert "tool.lookup_order" in all_operations, (
            f"tool.lookup_order specifically missing — the fn attached "
            f"to the agent did not emit a span. Operations seen: "
            f"{sorted(all_operations)}"
        )
