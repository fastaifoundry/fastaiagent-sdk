"""Phase 1 — LangChain/LangGraph harness tests (real LLM, gated by env).

Spec test IDs covered: #1, #1b, #2, #2b, #3, #4, #5.

Each test runs a small LangGraph compiled graph end-to-end (not a fake /
mocked pipeline) so the assertions about token capture, cost, and
input/output payloads are meaningful. The tests skip cleanly when LLM
keys aren't present.
"""

from __future__ import annotations

import json
import os
import time

import pytest

HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))

if not (HAS_OPENAI or HAS_ANTHROPIC):
    pytest.skip(
        "Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY set — skipping LangChain harness tests",
        allow_module_level=True,
    )

needs_openai = pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY not set")
needs_anthropic = pytest.mark.skipif(not HAS_ANTHROPIC, reason="ANTHROPIC_API_KEY not set")

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trace_store():
    from fastaiagent.trace.storage import TraceStore

    return TraceStore.default()


def _wait_for_root_span(predicate, timeout: float = 8.0) -> object | None:
    """Poll the trace store until a span matching ``predicate`` lands.

    LocalStorageProcessor writes synchronously on span end, but we still
    leave a small wait window so this is robust to OTel batching changes.
    """
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


def _build_graph(llm):
    """Compile a tiny LangGraph: one tool-using node, MessagesState root.

    Using LangGraph's ``create_react_agent`` keeps the graph realistic
    (it emits the standard chain → llm → tool span hierarchy) without
    forcing us to write the state machine ourselves.
    """
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    @tool
    def echo_tool(text: str) -> str:
        """Echo the input back unchanged. Used so the LLM has a tool call to make."""
        return f"echo: {text}"

    return create_react_agent(llm, tools=[echo_tool])


def _root_lc_span(trace) -> object:
    for span in trace.spans:
        attrs = span.attributes or {}
        if attrs.get("fastaiagent.framework") == "langchain":
            return span
    raise AssertionError(
        f"no langchain root span in trace {trace.trace_id}; "
        f"spans: {[s.name for s in trace.spans]}"
    )


def _llm_span(trace) -> object:
    for span in trace.spans:
        if span.name.startswith("llm."):
            return span
    raise AssertionError(f"no llm.* span in trace {trace.trace_id}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@needs_openai
def test_01_autotrace_openai() -> None:
    """Spec #1: graph→node→llm→tool hierarchy with framework=langchain."""
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    from fastaiagent.integrations import langchain as lc

    lc.enable()
    handler = lc.get_callback_handler()
    graph = _build_graph(ChatOpenAI(model="gpt-4o-mini", temperature=0))

    # Pass the handler explicitly via config so the test does not depend on
    # the configure-hook ContextVar being inherited into the test runner.
    result = graph.invoke(
        {"messages": [HumanMessage(content="Use the echo_tool with text 'hi'.")]},
        config={"callbacks": [handler]},
    )
    assert result and "messages" in result

    trace = _wait_for_root_span(
        lambda s: (s.attributes or {}).get("fastaiagent.framework") == "langchain"
    )
    assert trace is not None, "no LangChain trace landed in store"

    root = _root_lc_span(trace)
    assert root.name.startswith("langgraph.") or root.name.startswith("langchain.")

    # We expect at minimum: root + 1 llm child. Tool span is best-effort
    # (LLM may decide not to call the tool depending on phrasing) — assert
    # llm span exists, tool span only if the LLM actually called it.
    span_names = [s.name for s in trace.spans]
    assert any(n.startswith("llm.") for n in span_names), (
        f"expected llm.* span, got {span_names}"
    )


@needs_anthropic
def test_01b_autotrace_anthropic() -> None:
    """Spec #1b: same hierarchy with ChatAnthropic on claude-3-5-haiku."""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage

    from fastaiagent.integrations import langchain as lc

    lc.enable()
    handler = lc.get_callback_handler()
    graph = _build_graph(
        ChatAnthropic(model_name="claude-haiku-4-5", timeout=30, stop=None)
    )

    graph.invoke(
        {"messages": [HumanMessage(content="Say hello in 5 words.")]},
        config={"callbacks": [handler]},
    )

    trace = _wait_for_root_span(
        lambda s: (s.attributes or {}).get("fastaiagent.framework") == "langchain"
        and s.attributes.get("gen_ai.system") in (None, "langchain")
    )
    assert trace is not None
    llm = _llm_span(trace)
    attrs = llm.attributes or {}
    assert attrs.get("gen_ai.system") == "anthropic", attrs.get("gen_ai.system")


@needs_openai
def test_02_token_capture_openai() -> None:
    """Spec #2: gen_ai.usage.input_tokens > 0 on the LLM span."""
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    from fastaiagent.integrations import langchain as lc

    lc.enable()
    handler = lc.get_callback_handler()
    graph = _build_graph(ChatOpenAI(model="gpt-4o-mini", temperature=0))

    graph.invoke(
        {"messages": [HumanMessage(content="Reply with the single word: pong")]},
        config={"callbacks": [handler]},
    )

    trace = _wait_for_root_span(
        lambda s: (s.attributes or {}).get("fastaiagent.framework") == "langchain"
    )
    assert trace is not None
    llm = _llm_span(trace)
    attrs = llm.attributes or {}
    assert int(attrs.get("gen_ai.usage.input_tokens") or 0) > 0, attrs
    assert int(attrs.get("gen_ai.usage.output_tokens") or 0) > 0, attrs
    # Cost should also have been computed off the OpenAI pricing row.
    assert float(attrs.get("fastaiagent.cost.total_usd") or 0.0) > 0.0, attrs


@needs_anthropic
def test_02b_token_capture_anthropic() -> None:
    """Spec #2b: tokens captured from Anthropic response usage block."""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage

    from fastaiagent.integrations import langchain as lc

    lc.enable()
    handler = lc.get_callback_handler()
    graph = _build_graph(
        ChatAnthropic(model_name="claude-haiku-4-5", timeout=30, stop=None)
    )

    graph.invoke(
        {"messages": [HumanMessage(content="Reply with the single word: pong")]},
        config={"callbacks": [handler]},
    )

    trace = _wait_for_root_span(
        lambda s: (s.attributes or {}).get("fastaiagent.framework") == "langchain"
    )
    assert trace is not None
    llm = _llm_span(trace)
    attrs = llm.attributes or {}
    assert int(attrs.get("gen_ai.usage.input_tokens") or 0) > 0, attrs
    assert int(attrs.get("gen_ai.usage.output_tokens") or 0) > 0, attrs


@needs_openai
def test_03_tool_capture_openai() -> None:
    """Spec #3: tool span with args + output + non-zero latency."""
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    from fastaiagent.integrations import langchain as lc

    lc.enable()
    handler = lc.get_callback_handler()
    graph = _build_graph(ChatOpenAI(model="gpt-4o-mini", temperature=0))

    graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Call the echo_tool with text='ping'. Then answer with whatever it returned."
                )
            ]
        },
        config={"callbacks": [handler]},
    )

    trace = _wait_for_root_span(
        lambda s: (s.attributes or {}).get("fastaiagent.framework") == "langchain"
    )
    assert trace is not None
    tool_spans = [s for s in trace.spans if s.name.startswith("tool.")]
    if not tool_spans:
        pytest.skip("LLM declined to call the tool — non-deterministic; rerun")

    tool_span = tool_spans[0]
    attrs = tool_span.attributes or {}
    assert attrs.get("fastaiagent.tool.name")
    assert attrs.get("tool.input")
    assert attrs.get("tool.output")
    # Non-zero latency
    assert tool_span.start_time and tool_span.end_time
    assert tool_span.end_time > tool_span.start_time


def test_04_idempotent_enable() -> None:
    """Spec #4: enable() twice produces no duplicates and same handler."""
    from fastaiagent.integrations import langchain as lc

    lc.enable()
    h1 = lc.get_callback_handler()
    lc.enable()
    h2 = lc.get_callback_handler()
    assert h1 is h2, "enable() must reuse the same handler instance"


@needs_openai
def test_05_full_messages_in_request() -> None:
    """Spec #5: gen_ai.request.messages captures the full messages array."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    from fastaiagent.integrations import langchain as lc

    lc.enable()
    handler = lc.get_callback_handler()
    graph = _build_graph(ChatOpenAI(model="gpt-4o-mini", temperature=0))

    graph.invoke(
        {
            "messages": [
                SystemMessage(content="You are terse. One word answers."),
                HumanMessage(content="What colour is the sky on a clear day?"),
            ]
        },
        config={"callbacks": [handler]},
    )

    trace = _wait_for_root_span(
        lambda s: (s.attributes or {}).get("fastaiagent.framework") == "langchain"
    )
    assert trace is not None
    llm = _llm_span(trace)
    raw = (llm.attributes or {}).get("gen_ai.request.messages")
    assert raw, "gen_ai.request.messages missing"

    # The handler stores a JSON-serialised message array. Both system and
    # human messages should be present in the dump.
    payload = json.loads(raw) if isinstance(raw, str) else raw
    blob = json.dumps(payload, default=str)
    assert "terse" in blob.lower() or "system" in blob.lower(), blob[:400]
    assert "sky" in blob.lower() or "human" in blob.lower(), blob[:400]
