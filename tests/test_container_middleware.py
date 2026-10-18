"""``middleware=`` on Chain, Swarm and Supervisor; middleware order in traces (1.91.0, 1C).

Before 1.91.0 only ``Agent`` took middleware, so a policy that should hold across a
whole team — a tool budget, an audit log, a redactor — had to be copied onto every
agent by hand, and a Chain's tool nodes could not be wrapped at all (X7). Now:

* ``Chain(middleware=...)`` runs outermost around every **agent node** (outside
  the node agent's own middleware, without changing that agent) and wraps every
  **tool node** in ``wrap_tool``. A chain without middleware runs exactly as before.
* ``Swarm(middleware=...)`` applies to every hop, outside each agent's own.
* ``Supervisor(middleware=...)`` applies to the supervisor's own loop (its
  ``delegate_to_*`` calls); workers keep theirs. With ``routing="decisions"`` there
  is no supervisor loop, so it applies to the routed worker's run.
* A container's middleware keeps ONE ``ctx.state`` for the container's run — a
  swarm-level ``ToolBudget`` counts every hop's tools — and a Swarm resume picks it
  up again.
* The run's root span carries ``fastaiagent.middleware``: the middleware in order
  (decision B1-6 — one structural attribute, no new spans).

Real ``Chain`` / ``Swarm`` / ``Supervisor`` / ``Agent``, real checkpointer, real OTel
spans written to a private ``local.db``; the model is the repo's ``MockLLMClient``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from fastaiagent import (
    Agent,
    AgentMiddleware,
    FunctionTool,
    SQLiteCheckpointer,
    Supervisor,
    Swarm,
    ToolBudget,
    Worker,
)
from fastaiagent.chain import Chain, NodeType
from fastaiagent.chain.interrupt import Resume, interrupt
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import ToolCall
from fastaiagent.tool.base import ToolResult
from tests.conftest import MockLLMClient


def echo(text: str) -> str:
    """Echo the text."""
    return f"echo:{text}"


def double(value: int) -> dict:
    """Double a number."""
    return {"doubled": int(value) * 2}


def _approve(text: str) -> str:
    """Needs approval."""
    interrupt(reason="approve?", context={"text": text})
    return f"approved:{text}"


def _call(name: str, cid: str, **args: Any) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=cid, name=name, arguments=args)],
        finish_reason="tool_calls",
    )


def _answer(text: str = "done") -> LLMResponse:
    return LLMResponse(content=text, finish_reason="stop")


class Log(AgentMiddleware):
    def __init__(self, name: str, log: list[str]) -> None:
        self.name, self.log = name, log

    async def before_run(self, ctx, input):
        self.log.append(f"{self.name}:run")
        return input

    async def wrap_tool(self, ctx, tool, args, call_next):
        self.log.append(f"{self.name}:tool:{tool.name}")
        return await call_next(tool, args)


# ── Chain ──────────────────────────────────────────────────────────────────


def _chain(middleware: list[AgentMiddleware] | None, agent: Agent) -> Chain:
    chain = Chain("c", checkpoint_enabled=False, middleware=middleware)
    chain.add_node(
        "calc",
        tool=FunctionTool(name="double", fn=double),
        type=NodeType.tool,
        input_mapping={"value": "{{state.n}}"},
    )
    chain.add_node("talk", agent=agent)
    chain.connect("calc", "talk")
    return chain


@pytest.mark.asyncio
async def test_chain_middleware_wraps_agent_nodes_outermost_and_tool_nodes() -> None:
    log: list[str] = []
    agent = Agent(
        name="talker",
        llm=MockLLMClient(responses=[_call("echo", "c1", text="hi"), _answer()]),
        tools=[FunctionTool(name="echo", fn=echo)],
        middleware=[Log("inner", log)],
    )
    await _chain([Log("outer", log)], agent).aexecute({"n": 2})
    assert log == [
        "outer:tool:double",  # the tool node
        "outer:run",  # the agent node: chain middleware first …
        "inner:run",
        "outer:tool:echo",  # … and outermost around the agent's tools
        "inner:tool:echo",
    ]
    assert [type(m) for m in agent.middleware] == [Log]  # the node agent is unchanged


@pytest.mark.asyncio
async def test_chain_tool_node_can_be_short_circuited_by_middleware() -> None:
    class Cached(AgentMiddleware):
        async def wrap_tool(self, ctx, tool, args, call_next):
            return ToolResult(output={"doubled": 99})

    agent = Agent(name="talker", llm=MockLLMClient(responses=[_answer()]))
    result = await _chain([Cached()], agent).aexecute({"n": 2})
    assert result.node_results["calc"]["output"] == {"doubled": 99}


@pytest.mark.asyncio
async def test_a_chain_without_middleware_runs_its_tool_nodes_as_before() -> None:
    agent = Agent(name="talker", llm=MockLLMClient(responses=[_answer()]))
    result = await _chain(None, agent).aexecute({"n": 2})
    assert result.node_results["calc"]["output"] == {"doubled": 4}


@pytest.mark.asyncio
async def test_a_chain_budget_counts_every_node() -> None:
    """One ToolBudget for the chain's run: the tool node uses 1, the agent node's
    first tool call uses the 2nd, its second call is over budget."""
    agent = Agent(
        name="talker",
        llm=MockLLMClient(
            responses=[_call("echo", "c1", text="a"), _call("echo", "c2", text="b"), _answer()]
        ),
        tools=[FunctionTool(name="echo", fn=echo)],
    )
    result = await _chain([ToolBudget(max_calls=2)], agent).aexecute({"n": 2})
    assert result.node_results["talk"]["output"] == "Tool budget exhausted."


# ── Swarm ──────────────────────────────────────────────────────────────────


def _handoff_pair(llm_a: MockLLMClient, llm_b: MockLLMClient, **kw: Any) -> list[Agent]:
    a = Agent(name="a", llm=llm_a, tools=[FunctionTool(name="echo", fn=echo)], **kw.pop("a_kw", {}))
    b = Agent(
        name="b",
        llm=llm_b,
        tools=[FunctionTool(name="echo", fn=echo), FunctionTool(name="approve", fn=_approve)],
    )
    return [a, b]


@pytest.mark.asyncio
async def test_swarm_middleware_runs_on_every_hop_outside_the_agents_own() -> None:
    log: list[str] = []
    llm_a = MockLLMClient(responses=[_call("handoff_to_b", "h1", reason="b knows")])
    llm_b = MockLLMClient(responses=[_call("echo", "c1", text="x"), _answer("b done")])
    agents = _handoff_pair(llm_a, llm_b, a_kw={"middleware": [Log("own", log)]})
    swarm = Swarm(
        name="s",
        agents=agents,
        entrypoint="a",
        handoffs={"a": ["b"], "b": []},
        middleware=[Log("team", log)],
    )
    result = await swarm.arun("go")
    assert result.output == "b done"
    assert log[:2] == ["team:run", "own:run"]  # hop a: team outside own
    assert "team:tool:handoff_to_b" in log and "team:tool:echo" in log
    assert log.count("team:run") == 2  # every hop


@pytest.mark.asyncio
async def test_a_swarm_budget_counts_across_hops_and_survives_a_resume(tmp_path) -> None:
    store = SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))
    llm_a = MockLLMClient(
        responses=[_call("echo", "a1", text="1"), _call("handoff_to_b", "h1", reason="b")]
    )
    llm_b = MockLLMClient(
        responses=[
            _call("approve", "b1", text="2"),  # pauses — budget 3 of 4 (incl. the handoff)
            _call("echo", "b2", text="3"),  # 4 of 4
            _call("echo", "b3", text="4"),  # over
            _answer("should not get here"),
        ]
    )
    swarm = Swarm(
        name="s",
        agents=_handoff_pair(llm_a, llm_b),
        entrypoint="a",
        handoffs={"a": ["b"], "b": []},
        middleware=[ToolBudget(max_calls=4)],
        checkpointer=store,
    )
    paused = await swarm.arun("go", execution_id="sw")
    assert paused.status == "paused"
    done = await swarm.aresume("sw", resume_value=Resume(approved=True))
    assert done.output == "Tool budget exhausted."


@pytest.mark.asyncio
async def test_a_swarm_budget_counts_across_hops_on_astream() -> None:
    llm_a = MockLLMClient(
        responses=[_call("echo", "a1", text="1"), _call("handoff_to_b", "h1", reason="b")]
    )
    llm_b = MockLLMClient(
        responses=[_call("echo", "b1", text="2"), _call("echo", "b2", text="3"), _answer("no")]
    )
    swarm = Swarm(
        name="s",
        agents=_handoff_pair(llm_a, llm_b),
        entrypoint="a",
        handoffs={"a": ["b"], "b": []},
        # On a stream the handoff is not run as a tool, so hop a uses 1, b1 the
        # 2nd, and b2 is over — only if the count carried from hop a.
        middleware=[ToolBudget(max_calls=2)],
    )
    from fastaiagent.llm.stream import TextDelta

    text = "".join([e.text async for e in swarm.astream("go") if isinstance(e, TextDelta)])
    assert text.endswith("Tool budget exhausted.")


def test_swarm_stream_returns_the_same_answer_as_run() -> None:
    """X14: ``Swarm.stream()`` joined every delta, so what an agent said before
    handing off became part of the swarm's answer, unlike ``Swarm.run()``."""

    def swarm() -> Swarm:
        a = Agent(
            name="a",
            llm=MockLLMClient(
                responses=[
                    LLMResponse(
                        content="Passing you to b. ",
                        tool_calls=[
                            ToolCall(id="h1", name="handoff_to_b", arguments={"reason": "b"})
                        ],
                        finish_reason="tool_calls",
                    )
                ]
            ),
        )
        b = Agent(name="b", llm=MockLLMClient(responses=[_answer("The answer is 4.")]))
        return Swarm(name="s", agents=[a, b], entrypoint="a", handoffs={"a": ["b"], "b": []})

    assert swarm().run("q").output == "The answer is 4."
    assert swarm().stream("q").output == "The answer is 4."


@pytest.mark.asyncio
async def test_a_tool_name_clash_with_team_middleware_raises_at_construction() -> None:
    class Brings(AgentMiddleware):
        name = "brings"
        tools = [FunctionTool(name="echo", fn=echo)]

    agent = Agent(name="a", llm=MockLLMClient(), tools=[FunctionTool(name="echo", fn=echo)])
    with pytest.raises(ValueError, match="echo"):
        Swarm(name="s", agents=[agent], entrypoint="a", middleware=[Brings()])


# ── Supervisor ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_supervisor_middleware_runs_on_the_routed_worker() -> None:
    log: list[str] = []
    worker = Agent(
        name="w",
        llm=MockLLMClient(responses=[_call("echo", "w1", text="x"), _answer("worker done")]),
        tools=[FunctionTool(name="echo", fn=echo)],
    )
    from fastaiagent.testing import TestModel

    router = TestModel(
        decisions={
            "answers": [{"type": "choice", "name": "worker", "choice": "w", "confidence": 0.9}]
        }
    )
    idle = Agent(name="idle", llm=MockLLMClient(responses=[_answer("idle")]))
    sup = Supervisor(
        name="lead",
        workers=[
            Worker(agent=worker, description="does things"),
            Worker(agent=idle, description="does nothing"),
        ],
        routing="decisions",
        router_llm=router,
        fallback_worker="idle",
        middleware=[Log("sup", log)],
    )
    result = await sup.arun("go")
    assert result.output == "worker done"
    assert log == ["sup:run", "sup:tool:echo"]
    assert [type(m) for m in worker.middleware] == []  # the user's worker is unchanged


@pytest.mark.asyncio
async def test_supervisor_middleware_wraps_the_supervisor_loop_not_the_workers() -> None:
    log: list[str] = []
    worker = Agent(
        name="w",
        llm=MockLLMClient(responses=[_call("echo", "w1", text="x"), _answer("worker done")]),
        tools=[FunctionTool(name="echo", fn=echo)],
    )
    lead = MockLLMClient(
        responses=[_call("delegate_to_w", "s1", task="do it"), _answer("lead done")]
    )
    sup = Supervisor(
        name="lead", llm=lead, workers=[Worker(agent=worker)], middleware=[Log("sup", log)]
    )
    result = await sup.arun("go")
    assert result.output == "lead done"
    assert log == ["sup:run", "sup:tool:delegate_to_w"]


# ── the order in traces ────────────────────────────────────────────────────


@pytest.fixture
def traced(monkeypatch, tmp_path) -> Iterator[str]:
    from fastaiagent._internal import instance as _instance
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    reset_config()
    _project.set_project_id("test-proj")
    _instance.reset_for_testing()
    otel.reset()
    try:
        yield str(db)
    finally:
        otel.reset()
        _instance.reset_for_testing()
        _project.reset_for_testing()
        reset_config()


def _root(db: str, trace_id: str) -> dict[str, Any]:
    from fastaiagent.trace.otel import get_tracer_provider
    from fastaiagent.trace.storage import TraceStore

    provider = get_tracer_provider()
    if provider is not None and hasattr(provider, "force_flush"):
        provider.force_flush()
    spans = TraceStore(db).get_trace(trace_id).spans
    return next(s for s in spans if not s.parent_span_id).attributes


@pytest.mark.parametrize("mode", ["run", "stream"])
def test_the_agent_root_span_names_its_middleware_in_order(mode: str, traced: str) -> None:
    agent = Agent(
        name="m",
        llm=MockLLMClient(responses=[_answer()]),
        middleware=[Log("first", []), ToolBudget(max_calls=3)],
    )
    result = agent.run("q") if mode == "run" else agent.stream("q")
    assert list(_root(traced, result.trace_id)["fastaiagent.middleware"]) == [
        "first",
        "tool_budget",
    ]


def test_no_middleware_means_no_attribute(traced: str) -> None:
    result = Agent(name="m", llm=MockLLMClient(responses=[_answer()])).run("q")
    assert "fastaiagent.middleware" not in _root(traced, result.trace_id)


def test_container_root_spans_name_their_middleware(traced: str) -> None:
    worker = Agent(name="w", llm=MockLLMClient(responses=[_answer("worker done")]))
    lead = MockLLMClient(responses=[_call("delegate_to_w", "s1", task="t"), _answer("lead")])
    sup = Supervisor(
        name="lead", llm=lead, workers=[Worker(agent=worker)], middleware=[ToolBudget()]
    )
    result = sup.run("go")
    assert list(_root(traced, result.trace_id)["fastaiagent.middleware"]) == ["tool_budget"]
