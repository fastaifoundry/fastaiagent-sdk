"""One team run is one trace, on every run path (1.99.0).

Swept across Supervisor (tools routing) / Supervisor (decisions routing) / Swarm ×
``run`` / ``arun`` / ``stream`` / ``astream``: exactly one root span, named for the
team, every agent span under it, and the team's span exactly once.

What it caught, live against the plane:
* ``Swarm.astream()`` opened no root span — one streamed run was one unrelated
  trace per hop;
* ``Supervisor.astream()`` with ``routing="decisions"`` opened none either;
* ``Supervisor.stream()`` nested a second ``supervisor.<name>`` inside its own;
* ``swarm.handoff_count`` counted every tool call (a run with 3 handoffs and 6 KB
  searches said 9).

No mocks: real OTel spans through the SDK's provider; the models are the offline
``FunctionModel`` / ``TestModel`` from ``fastaiagent.testing``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

import fastaiagent as fa
from fastaiagent.testing import FunctionModel, TestModel
from fastaiagent.tool import FunctionTool
from fastaiagent.trace.otel import get_tracer_provider


class _Collector(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[Any] = []

    def export(self, spans):  # type: ignore[override]
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover
        pass


@pytest.fixture()
def collector() -> _Collector:
    col = _Collector()
    get_tracer_provider().add_span_processor(SimpleSpanProcessor(col))
    return col


def _lead_route(messages: list[Any]) -> Any:
    if messages[-1].role.value == "tool":
        return "team says done"
    return "", [{"name": "delegate_to_a", "arguments": {"task": "help"}}]


def _searching_entry(messages: list[Any]) -> Any:
    """Search once, then hand off — a run with one handoff and two tool calls."""
    if messages[-1].role.value == "tool":
        return "", [{"name": "handoff_to_b", "arguments": {"reason": "yours"}}]
    return "", [{"name": "search", "arguments": {"q": "x"}}]


def _team(topology: str) -> Any:
    if topology == "swarm":
        a = fa.Agent(
            name="a",
            llm=FunctionModel(_searching_entry, model="a-model"),
            tools=[FunctionTool(name="search", fn=lambda q: "found")],
        )
        b = fa.Agent(name="b", llm=TestModel("answer", model="b-model"))
        return fa.Swarm(name="tree-swarm", agents=[a, b], entrypoint="a")
    workers = [
        fa.Worker(fa.Agent(name="a", llm=TestModel("worker a", model="a-model")), role="a"),
        fa.Worker(fa.Agent(name="b", llm=TestModel("worker b", model="b-model")), role="b"),
    ]
    if topology == "supervisor":
        return fa.Supervisor(name="tree-lead", llm=FunctionModel(_lead_route), workers=workers)
    router = TestModel(
        decisions={
            "answers": [
                {
                    "type": "choice",
                    "name": "worker",
                    "choice": "a",
                    "probabilities": [
                        {"value": "a", "probability": 0.9},
                        {"value": "b", "probability": 0.1},
                    ],
                    "confidence": 0.9,
                }
            ]
        }
    )
    return fa.Supervisor(
        name="tree-desk",
        llm=TestModel("unused"),
        workers=workers,
        routing="decisions",
        router_llm=router,
        fallback_worker="b",
    )


def _drive(team: Any, path: str) -> None:
    if path == "run":
        team.run("hi")
    elif path == "arun":
        asyncio.run(team.arun("hi"))
    elif path == "stream":
        team.stream("hi")
    else:

        async def consume() -> None:
            async for _ in team.astream("hi"):
                pass

        asyncio.run(consume())


ROOT = {"supervisor": "supervisor.tree-lead", "decisions": "supervisor.tree-desk",
        "swarm": "swarm.tree-swarm"}  # fmt: skip


@pytest.mark.parametrize("path", ["run", "arun", "stream", "astream"])
@pytest.mark.parametrize("topology", ["supervisor", "decisions", "swarm"])
def test_one_team_run_is_one_trace(topology: str, path: str, collector: _Collector) -> None:
    _drive(_team(topology), path)

    spans = collector.spans
    roots = [s for s in spans if s.parent is None]
    assert [s.name for s in roots] == [ROOT[topology]], [s.name for s in roots]
    assert {s.context.trace_id for s in spans} == {roots[0].context.trace_id}
    assert [s.name for s in spans].count(ROOT[topology]) == 1
    assert any(s.name.startswith("agent.") for s in spans)


@pytest.mark.parametrize("path", ["run", "arun", "stream", "astream"])
def test_swarm_handoff_count_counts_handoffs(path: str, collector: _Collector) -> None:
    _drive(_team("swarm"), path)

    root = next(s for s in collector.spans if s.name == ROOT["swarm"])
    assert root.attributes["swarm.handoff_count"] == 1  # not 2: the search is not a handoff
