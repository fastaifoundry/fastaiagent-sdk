"""Live: LangChain 1.x ``create_agent`` traces like ``create_react_agent`` (1.89.0, WS6).

The ``[langchain]`` extra admits 1.x from 1.89.0, and 1.x's builder is
``langchain.agents.create_agent``. The existing harness modules stay on
``create_react_agent`` (which works on both lines, so 0.x keeps live coverage);
this one covers the 1.x builder, plus every streamed path whose cost was
missing before 1.89.0 (``stream_mode="messages"`` and the v3 protocol).

Each test finds its own trace through a unique ``agent_name(...)``: never
"any LangChain trace", which would certify rather than check.

On LangChain 0.x the whole module skips with the line reason the
``langchain-lines`` CI job allows. On 1.x a missing ``create_agent`` is an
error, not a skip.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

pytest.importorskip("langchain_core")

import langchain_core  # noqa: E402

_CORE_MAJOR = int(langchain_core.__version__.split(".")[0])
if _CORE_MAJOR < 1:
    pytest.skip(
        f"not on this LangChain line: create_agent is 1.x-only (core {langchain_core.__version__})",
        allow_module_level=True,
    )

from langchain.agents import create_agent  # noqa: E402  (raises on 1.x if missing)

HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
needs_openai = pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY not set")
needs_anthropic = pytest.mark.skipif(not HAS_ANTHROPIC, reason="ANTHROPIC_API_KEY not set")

pytestmark = pytest.mark.e2e


@pytest.fixture
def name() -> str:
    return f"ca-{uuid.uuid4().hex[:8]}"


def _graph(llm: Any) -> Any:
    from langchain_core.tools import tool

    @tool
    def echo_tool(text: str) -> str:
        """Echo the input back unchanged."""
        return f"echo: {text}"

    return create_agent(llm, tools=[echo_tool])


def _openai() -> Any:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="gpt-5-mini", reasoning_effort="minimal")


def _trace_named(agent: str) -> Any:
    from fastaiagent.trace.storage import TraceStore

    store = TraceStore.default()
    for summary in store.list_traces(last_hours=None, limit=None):
        trace = store.get_trace(summary.trace_id)
        if any((s.attributes or {}).get("agent.name") == agent for s in trace.spans):
            return trace
    raise AssertionError(f"no trace stamped agent.name={agent!r}")


def _llm_spans(trace: Any) -> list[Any]:
    spans = [s for s in trace.spans if s.name.startswith("llm.")]
    assert spans, f"no llm.* span in {[s.name for s in trace.spans]}"
    return spans


def _assert_priced(trace: Any) -> None:
    for span in _llm_spans(trace):
        attrs = span.attributes or {}
        assert attrs.get("gen_ai.usage.input_tokens"), f"{span.name}: no tokens"
        cost = attrs.get("fastaiagent.cost.total_usd")
        assert cost is not None and cost > 0, f"{span.name}: no SDK cost ({attrs})"


def _say(text: str) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage

    return {"messages": [HumanMessage(content=text)]}


@needs_openai
def test_trace_shape_on_invoke(name: str) -> None:
    from fastaiagent.integrations import agent_name
    from fastaiagent.integrations import langchain as lc

    lc.enable()
    graph = _graph(_openai())
    with agent_name(name):
        graph.invoke(
            _say("Use echo_tool on 'hi', then reply done."),
            config={"callbacks": [lc.get_callback_handler()]},
        )

    trace = _trace_named(name)
    ids = {s.span_id for s in trace.spans}
    orphans = [s.name for s in trace.spans if s.parent_span_id and s.parent_span_id not in ids]
    assert not orphans, f"orphan spans: {orphans}"
    root = next(s for s in trace.spans if s.parent_span_id is None)
    assert root.name.startswith("langchain.")
    assert (root.attributes or {}).get("fastaiagent.framework") == "langchain"
    for span in _llm_spans(trace):
        assert span.name.startswith("llm.openai."), span.name
        assert (span.attributes or {}).get("gen_ai.system") == "openai"
    _assert_priced(trace)


@needs_anthropic
def test_anthropic_tokens_and_cost(name: str) -> None:
    from langchain_anthropic import ChatAnthropic

    from fastaiagent.integrations import agent_name
    from fastaiagent.integrations import langchain as lc

    lc.enable()
    graph = _graph(ChatAnthropic(model="claude-haiku-4-5", temperature=0))
    with agent_name(name):
        graph.invoke(
            _say("Reply with the word ok."), config={"callbacks": [lc.get_callback_handler()]}
        )
    _assert_priced(_trace_named(name))


@needs_openai
def test_with_guardrails_blocks_input_pii(name: str) -> None:
    from fastaiagent.guardrail.builtins import no_pii
    from fastaiagent.guardrail.guardrail import GuardrailPosition
    from fastaiagent.integrations import langchain as lc
    from fastaiagent.integrations._registry import GuardrailBlocked

    lc.enable()
    guarded = lc.with_guardrails(
        _graph(_openai()), name=name, input_guardrails=[no_pii(GuardrailPosition.input)]
    )
    with pytest.raises(GuardrailBlocked):
        guarded.invoke(_say("My email is a@b.com"))


@needs_openai
def test_register_agent_topology(name: str) -> None:
    from fastaiagent.integrations import langchain as lc
    from fastaiagent.integrations._registry import fetch_agent

    lc.register_agent(_graph(_openai()), name=name)
    row = fetch_agent(name)
    assert row is not None and row["framework"] == "langchain"
    node_ids = {str(n.get("id")) for n in (row.get("topology") or {}).get("nodes") or []}
    assert {"model", "tools"} <= node_ids, node_ids


@needs_openai
def test_as_evaluable_returns_plain_text_and_a_trace(name: str) -> None:
    from fastaiagent.integrations import agent_name
    from fastaiagent.integrations import langchain as lc

    lc.enable()
    evaluable = lc.as_evaluable(_graph(_openai()))
    with agent_name(name):
        result = evaluable("What is the capital of France? One word.")
    assert isinstance(result.output, str) and "paris" in result.output.lower()
    assert "'type'" not in result.output, "content blocks came back as a repr"
    assert result.trace_id


@needs_openai
def test_stream_mode_messages_is_priced(name: str) -> None:
    """Streamed through LangGraph: no ``llm_output``, so no cost before 1.89.0."""
    from fastaiagent.integrations import agent_name
    from fastaiagent.integrations import langchain as lc

    lc.enable()
    graph = _graph(_openai())
    with agent_name(name):
        for _ in graph.stream(
            _say("Reply with the word ok."),
            config={"callbacks": [lc.get_callback_handler()]},
            stream_mode="messages",
        ):
            pass
    _assert_priced(_trace_named(name))


@needs_openai
def test_graph_v3_stream_events_is_priced(name: str) -> None:
    import importlib.metadata

    from fastaiagent.integrations import agent_name
    from fastaiagent.integrations import langchain as lc

    lg = importlib.metadata.version("langgraph")
    if tuple(int(p) for p in lg.split(".")[:2]) < (1, 2):
        pytest.skip(f"not on this LangChain line: v3 graph streaming needs langgraph>=1.2 ({lg})")

    lc.enable()
    graph = _graph(_openai())
    with agent_name(name):
        run = graph.stream_events(
            _say("Reply with the word ok."),
            config={"callbacks": [lc.get_callback_handler()]},
            version="v3",
        )
        assert run.output is not None
    _assert_priced(_trace_named(name))


@needs_openai
def test_chat_model_v3_stream_events_is_priced(name: str) -> None:
    from langchain_core.messages import HumanMessage

    from fastaiagent.integrations import agent_name
    from fastaiagent.integrations import langchain as lc

    lc.enable()
    model = _openai()
    with agent_name(name):
        stream = model.stream_events(
            [HumanMessage(content=f"Reply with the word ok. ({name})")],
            config={"callbacks": [lc.get_callback_handler()]},
            version="v3",
        )
        assert stream.output is not None
    from fastaiagent.trace.storage import TraceStore

    # A bare model call has no root agent span, so find ITS llm span by the
    # unique marker in the request, never "the last llm span in the store".
    store = TraceStore.default()
    spans = [
        s
        for summary in store.list_traces(last_hours=None, limit=None)
        for s in store.get_trace(summary.trace_id).spans
        if s.name.startswith("llm.")
        and name in str((s.attributes or {}).get("gen_ai.request.messages", ""))
    ]
    assert len(spans) == 1, f"expected this call's one llm span, found {len(spans)}"
    attrs = spans[0].attributes or {}
    cost = attrs.get("fastaiagent.cost.total_usd")
    assert cost is not None and cost > 0, attrs
