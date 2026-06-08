"""Tool-call spans emit the replay-safety class + tool span-type (wire contract).

Drives the REAL executor (``_invoke_tool_with_span``) through the SDK's own
tracer provider into a temp SQLite DB, then reads the stored span back and
asserts the two attributes the central Replay engine consumes:

* ``fastaiagent.tool.replay_class`` — the resolved class (default ``side_effecting``);
* ``fastaiagent.runner.type == "tool"`` — so the span is classified as a tool
  span. Normalization is OFF here (the default), which proves we set this
  explicitly at emit time rather than relying on the opt-in OTel-capture
  inference (which the ``connect()`` push path never runs).

No mocks, no network: the tools are trivial Python callables.
"""

from __future__ import annotations

import pytest

from fastaiagent.agent.executor import _invoke_tool_with_span
from fastaiagent.tool import FunctionTool, tool
from fastaiagent.trace import otel
from fastaiagent.trace.storage import TraceStore

# The span types the Enterprise classifies as a tool span (it reads
# attributes["span_type"] or attributes["fastaiagent.runner.type"]).
_TOOL_SPAN_TYPES = {"tool", "tool_call", "worker_call"}


@pytest.fixture(autouse=True)
def _clean_tracer(isolated_local_db):
    """Rebuild the tracer provider per test so it binds to the temp DB.

    Depends on ``isolated_local_db`` so the env/config (temp SQLite path) is in
    place *before* we drop the provider singleton; the executor's own
    ``get_tracer()`` then lazily rebuilds a ``LocalStorageProcessor`` pointed at
    that temp DB.
    """
    otel.reset()
    yield
    otel.reset()


async def _emit_tool_span(tool_obj, tool_name, arguments):
    await _invoke_tool_with_span(
        tool=tool_obj,
        tool_name=tool_name,
        arguments=arguments,
        context=None,
        guardrails=None,
    )


def _tool_span_attrs(db_path):
    store = TraceStore(db_path=str(db_path))
    try:
        traces = store.list_traces()
        assert len(traces) == 1, f"expected one trace, got {len(traces)}"
        spans = store.get_trace(traces[0].trace_id).spans
        tool_spans = [s for s in spans if s.name.startswith("tool.")]
        assert len(tool_spans) == 1, f"expected one tool span, got {len(tool_spans)}"
        return tool_spans[0].attributes
    finally:
        store.close()


@pytest.mark.asyncio
async def test_marked_read_only_tool_emits_read_only(isolated_local_db):
    @tool(name="lookup", replay_class="read_only")
    def lookup(q: str) -> str:
        """A pure read."""
        return f"result:{q}"

    await _emit_tool_span(lookup, "lookup", {"q": "x"})

    attrs = _tool_span_attrs(isolated_local_db)
    assert attrs["fastaiagent.tool.replay_class"] == "read_only"
    assert attrs["fastaiagent.runner.type"] == "tool"
    assert attrs["fastaiagent.runner.type"] in _TOOL_SPAN_TYPES


@pytest.mark.asyncio
async def test_marked_idempotent_tool_emits_idempotent(isolated_local_db):
    def upsert(key: str) -> str:
        return f"upserted:{key}"

    t = FunctionTool(name="upsert", fn=upsert, replay_class="idempotent")
    await _emit_tool_span(t, "upsert", {"key": "k"})

    attrs = _tool_span_attrs(isolated_local_db)
    assert attrs["fastaiagent.tool.replay_class"] == "idempotent"
    assert attrs["fastaiagent.runner.type"] == "tool"


@pytest.mark.asyncio
async def test_unmarked_tool_emits_side_effecting(isolated_local_db):
    def act(x: int) -> int:
        return x + 1

    t = FunctionTool(name="act", fn=act)  # unmarked -> safe default
    await _emit_tool_span(t, "act", {"x": 1})

    attrs = _tool_span_attrs(isolated_local_db)
    assert attrs["fastaiagent.tool.replay_class"] == "side_effecting"
    assert attrs["fastaiagent.runner.type"] == "tool"
    assert attrs["fastaiagent.runner.type"] in _TOOL_SPAN_TYPES


@pytest.mark.asyncio
async def test_hallucinated_tool_span_classified_with_safe_default(isolated_local_db):
    # The LLM names a tool that isn't registered: tool is None. The span must
    # still be classified as a tool span and carry the safe side_effecting
    # default — otherwise central replay would see an unclassified span and
    # spuriously diverge.
    await _emit_tool_span(None, "ghost_tool", {"a": 1})

    attrs = _tool_span_attrs(isolated_local_db)
    assert attrs["fastaiagent.tool.replay_class"] == "side_effecting"
    assert attrs["fastaiagent.runner.type"] == "tool"
    assert attrs.get("tool.status") == "unknown"
