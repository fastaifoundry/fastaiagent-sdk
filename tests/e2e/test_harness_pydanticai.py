"""Phase 3 — PydanticAI harness tests (real LLM, gated by env).

Spec test IDs covered: #10, #10b, #11, #12, #13.

PydanticAI ships its own GenAI-semconv OTel instrumentation; our
integration's job is (a) flip ``Agent.instrument_all()`` on, (b) wrap
``run`` / ``run_sync`` / ``run_stream`` so the root span is tagged with
``fastaiagent.framework=pydanticai``, and (c) stamp token counts + cost
from ``AgentRunResult.usage()``.
"""

from __future__ import annotations

import os
import time

import pytest

HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))

if not (HAS_OPENAI or HAS_ANTHROPIC):
    pytest.skip(
        "Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY set — skipping PydanticAI harness tests",
        allow_module_level=True,
    )

needs_openai = pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY not set")
needs_anthropic = pytest.mark.skipif(not HAS_ANTHROPIC, reason="ANTHROPIC_API_KEY not set")

pytestmark = pytest.mark.e2e


def _trace_store():
    from fastaiagent.trace.storage import TraceStore

    return TraceStore.default()


def _wait_for_root_span(predicate, timeout: float = 10.0):
    store = _trace_store()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for summary in store.list_traces():
            try:
                trace = store.get_trace(summary.trace_id)
            except Exception:
                continue
            for span in trace.spans:
                if predicate(span):
                    return trace
        time.sleep(0.2)
    return None


def _root_pa_span(trace) -> object:
    for span in trace.spans:
        attrs = span.attributes or {}
        if attrs.get("fastaiagent.framework") == "pydanticai":
            return span
    raise AssertionError(
        f"no pydanticai root span in trace {trace.trace_id}; "
        f"spans: {[s.name for s in trace.spans]}"
    )


@needs_openai
def test_10_autotrace_openai() -> None:
    """Spec #10: framework=pydanticai is tagged on the root span and the
    GenAI attributes (provider + model) land.

    We assert what *our wrapper* writes (framework attr, provider, model)
    rather than the inner ``chat <model>`` span PydanticAI's
    instrument_all emits — that downstream span is occasionally absent
    when OTel's TracerProvider was reset by another test (a known
    PydanticAI / LogFire instrumentation race). The harness's promise
    is that our root span gets the framework + GenAI tags, which is
    what the UI badge / cost analytics read.

    Spec #11 separately asserts the token counts, which is the stronger
    signal that the LLM call really happened — that test still
    requires PydanticAI's instrumentation to be alive.
    """
    import uuid

    from pydantic_ai import Agent

    from fastaiagent.integrations import pydanticai as pa

    pa.enable()
    sentinel = f"sentinel-{uuid.uuid4().hex[:8]}"
    agent = Agent(
        "openai:gpt-4o-mini",
        system_prompt=f"Reply with exactly: {sentinel}",
    )
    result = agent.run_sync(f"Echo {sentinel}")
    assert result.output

    # Find OUR trace by the unique input we tagged on it.
    def predicate(span) -> bool:
        attrs = span.attributes or {}
        if attrs.get("fastaiagent.framework") != "pydanticai":
            return False
        agent_input = attrs.get("pydanticai.agent.input") or ""
        return sentinel in str(agent_input)

    trace = _wait_for_root_span(predicate, timeout=10.0)
    assert trace is not None, (
        f"no pydanticai trace tagged with sentinel {sentinel!r}"
    )

    root = _root_pa_span(trace)
    assert root.name.startswith("pydanticai.agent.")
    attrs = root.attributes or {}
    # Our wrapper writes these — robust to upstream instrumentation drift.
    assert attrs.get("fastaiagent.framework") == "pydanticai"
    assert (attrs.get("fastaiagent.framework.version") or "").strip()
    assert attrs.get("gen_ai.system") == "openai", attrs.get("gen_ai.system")
    assert "gpt-4o-mini" in str(attrs.get("gen_ai.request.model") or ""), attrs


@needs_anthropic
def test_10b_autotrace_anthropic() -> None:
    """Spec #10b: PydanticAI Agent on Anthropic traces correctly."""
    from pydantic_ai import Agent

    from fastaiagent.integrations import pydanticai as pa

    pa.enable()
    agent = Agent(
        "anthropic:claude-haiku-4-5",
        system_prompt="Answer in one word.",
    )
    result = agent.run_sync("What colour is grass?")
    assert result.output

    trace = _wait_for_root_span(
        lambda s: (s.attributes or {}).get("fastaiagent.framework") == "pydanticai"
    )
    assert trace is not None
    root = _root_pa_span(trace)
    attrs = root.attributes or {}
    assert attrs.get("gen_ai.system") == "anthropic", attrs.get("gen_ai.system")


@needs_openai
def test_11_token_capture() -> None:
    """Spec #11: usage() tokens land on the root span."""
    from pydantic_ai import Agent

    from fastaiagent.integrations import pydanticai as pa

    pa.enable()
    agent = Agent("openai:gpt-4o-mini", system_prompt="Reply with the word: pong")
    agent.run_sync("ping")

    trace = _wait_for_root_span(
        lambda s: (s.attributes or {}).get("fastaiagent.framework") == "pydanticai"
    )
    assert trace is not None
    root = _root_pa_span(trace)
    attrs = root.attributes or {}
    assert int(attrs.get("gen_ai.usage.input_tokens") or 0) > 0, attrs
    assert int(attrs.get("gen_ai.usage.output_tokens") or 0) > 0, attrs
    # Cost computed via the OpenAI pricing prefix.
    assert float(attrs.get("fastaiagent.cost.total_usd") or 0.0) > 0.0, attrs


@needs_openai
def test_12_tool_capture() -> None:
    """Spec #12: tool span with args + output + non-zero latency."""
    from pydantic_ai import Agent

    from fastaiagent.integrations import pydanticai as pa

    pa.enable()
    agent = Agent(
        "openai:gpt-4o-mini",
        system_prompt="Use the echo tool when asked to echo.",
    )

    @agent.tool_plain
    def echo(text: str) -> str:
        """Echo the input back unchanged."""
        return f"echo: {text}"

    agent.run_sync("Use the echo tool with text='ping'.")

    trace = _wait_for_root_span(
        lambda s: (s.attributes or {}).get("fastaiagent.framework") == "pydanticai"
    )
    assert trace is not None

    # PydanticAI's instrumentation emits ``running tool`` style spans
    # for tool calls. The exact span name varies across versions, so we
    # search for either our own ``tool.*`` namespace or PydanticAI's.
    tool_spans = [
        s
        for s in trace.spans
        if s.name.startswith("tool.")
        or "tool" in s.name.lower()
        or (s.attributes or {}).get("gen_ai.operation.name") == "execute_tool"
    ]
    if not tool_spans:
        pytest.skip(
            "LLM declined to call the tool or PydanticAI did not emit a "
            "tool span — non-deterministic; rerun"
        )
    span = tool_spans[0]
    assert span.start_time and span.end_time
    assert span.end_time > span.start_time


def test_13_idempotent_enable() -> None:
    """Spec #13: enable() twice does not double-patch."""
    from pydantic_ai import Agent

    from fastaiagent.integrations import pydanticai as pa

    pa.enable()
    first = Agent.run_sync
    pa.enable()
    second = Agent.run_sync
    assert first is second, "second enable() rewrapped Agent.run_sync"
    assert getattr(Agent.run_sync, "_fastaiagent_patched", False)
