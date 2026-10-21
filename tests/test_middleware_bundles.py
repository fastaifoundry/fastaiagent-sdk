"""A middleware brings its own tools, instructions and saved state (1.91.0, Bet 1 piece 1B).

A policy such as "refunds" is three things — the tools that act, the prompt text
that says when to use them, and the rules (a budget, a counter) that hold across
the run. Before 1.91.0 those lived in three places. Now one middleware carries all
of them:

* ``tools`` — merged into the run's tools (a name clash is an error at
  ``Agent(...)``, not a silent shadow);
* ``instructions`` — a string or ``callable(ctx)``; one system message right after
  the system prompt (before memory), so ``TrimLongMessages`` keeps it;
* ``ctx.state`` — a JSON-safe dict **per middleware**, saved in checkpoints under
  ``state_snapshot["middleware_state"]`` and restored on resume and fork.
  ``ToolBudget`` keeps its count there, so a budget now survives a resume
  (decision B1-5); before, it silently reset.

Real ``Agent``, real ``SQLiteCheckpointer``; the model is the repo's
``MockLLMClient`` (records the tools and messages of every call).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from fastaiagent import Agent, AgentMiddleware, FunctionTool, SQLiteCheckpointer
from fastaiagent.agent.middleware import ToolBudget, TrimLongMessages
from fastaiagent.chain.interrupt import Resume, interrupt
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import ToolCall
from tests.conftest import MockLLMClient


def echo(text: str) -> str:
    """Echo the text."""
    return f"echo:{text}"


def refund(order_id: str) -> str:
    """Refund an order."""
    return f"refunded {order_id}"


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


class Refunds(AgentMiddleware):
    name = "refunds"
    tools = [FunctionTool(name="refund", fn=refund)]
    instructions = "Refunds: only for orders under 30 days old. Use the refund tool."


def _offered(call: dict[str, Any]) -> list[str]:
    return [t["function"]["name"] for t in (call["tools"] or [])]


def _store(tmp_path, name: str = "cp.db") -> SQLiteCheckpointer:
    store = SQLiteCheckpointer(db_path=str(tmp_path / name))
    store.setup()
    return store


# ── tools ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["arun", "astream"])
@pytest.mark.asyncio
async def test_a_middlewares_tools_are_offered_and_callable(mode: str) -> None:
    llm = MockLLMClient(responses=[_call("refund", "c1", order_id="A-17"), _answer()])
    agent = Agent(
        name="b", llm=llm, tools=[FunctionTool(name="echo", fn=echo)], middleware=[Refunds()]
    )
    if mode == "arun":
        result = await agent.arun("refund A-17")
        assert result.tool_calls[0]["tool_name"] == "refund"
    else:
        [e async for e in agent.astream("refund A-17")]
    assert _offered(llm._calls[0]) == ["echo", "refund"]
    tool_results = [m.content for m in llm._calls[1]["messages"] if m.role.value == "tool"]
    assert tool_results == ["refunded A-17"]
    assert [t.name for t in agent.tools] == ["echo"]  # the agent's own list is untouched


def test_a_tool_name_clash_is_an_error_at_construction() -> None:
    with pytest.raises(ValueError, match="refund"):
        Agent(name="b", tools=[FunctionTool(name="refund", fn=echo)], middleware=[Refunds()])

    class MoreRefunds(Refunds):
        name = "more_refunds"

    with pytest.raises(ValueError, match="refund"):
        Agent(name="b", middleware=[Refunds(), MoreRefunds()])


@pytest.mark.parametrize("container", ["swarm", "supervisor"])
@pytest.mark.asyncio
async def test_middleware_tools_and_deferred_tools_inside_teams(container: str) -> None:
    """Swarm hops and Supervisor workers are rebuilt from the agent; its
    middleware's tools and instructions, and its deferred tools, come along."""
    from fastaiagent import Supervisor, Swarm, Worker

    def lookup(q: str) -> str:
        """Look something up."""
        return f"found:{q}"

    llm = MockLLMClient(
        responses=[
            _call("refund", "w1", order_id="A-17"),
            _call("load_tools", "w2", names=["lookup"]),
            _call("lookup", "w3", q="A-17"),
            _answer("worker done"),
        ]
    )
    agent = Agent(
        name="desk",
        llm=llm,
        tools=[FunctionTool(name="lookup", fn=lookup, deferred=True)],
        middleware=[Refunds()],
    )
    if container == "swarm":
        result = await Swarm(name="s", agents=[agent], entrypoint="desk").arun("refund A-17")
    else:
        lead = MockLLMClient(
            responses=[_call("delegate_to_desk", "s1", task="refund A-17"), _answer("lead done")]
        )
        result = await Supervisor(name="lead", llm=lead, workers=[Worker(agent=agent)]).arun("go")
    assert result.status == "completed"
    assert "refund" in _offered(llm._calls[0]) and "lookup" not in _offered(llm._calls[0])
    assert any(m.content == Refunds.instructions for m in llm._calls[0]["messages"])
    assert "lookup" in _offered(llm._calls[2])
    assert [m.content for m in llm._calls[3]["messages"] if m.role.value == "tool"][
        -1
    ] == "found:A-17"


# ── instructions ───────────────────────────────────────────────────────────


def test_instructions_follow_the_system_prompt() -> None:
    llm = MockLLMClient(responses=[_answer()])
    Agent(name="b", system_prompt="You are support.", llm=llm, middleware=[Refunds()]).run("hi")
    messages = llm._calls[0]["messages"]
    assert [m.role.value for m in messages] == ["system", "system", "user"]
    assert messages[0].content == "You are support."
    assert messages[1].content == Refunds.instructions


def test_callable_instructions_see_the_run_and_join_in_order() -> None:
    class Tier(AgentMiddleware):
        name = "tier"

        def instructions(self, ctx):
            return f"Customer tier: {ctx.run_context.state['tier']}."

    from fastaiagent import RunContext

    llm = MockLLMClient(responses=[_answer()])
    Agent(name="b", system_prompt="S", llm=llm, middleware=[Refunds(), Tier()]).run(
        "hi", context=RunContext(state={"tier": "gold"})
    )
    assert llm._calls[0]["messages"][1].content == (
        f"{Refunds.instructions}\n\nCustomer tier: gold."
    )


def test_trim_long_messages_keeps_the_instructions() -> None:
    llm = MockLLMClient(responses=[_answer()])
    from fastaiagent.llm.message import AssistantMessage, UserMessage

    history = []
    for i in range(10):
        history += [UserMessage(f"q{i}"), AssistantMessage(f"a{i}")]
    Agent(
        name="b", system_prompt="S", llm=llm, middleware=[Refunds(), TrimLongMessages(keep_last=3)]
    ).run("now", messages=history)
    sent = llm._calls[0]["messages"]
    assert sent[1].content == Refunds.instructions
    assert len([m for m in sent if m.role.value != "system"]) == 3


# ── ctx.state ──────────────────────────────────────────────────────────────


class Counter(AgentMiddleware):
    def __init__(self, name: str) -> None:
        self.name = name
        self.seen: list[int] = []

    async def before_model(self, ctx, messages):
        ctx.state["n"] = ctx.state.get("n", 0) + 1
        return messages

    async def wrap_tool(self, ctx, tool, args, call_next):
        before = dict(ctx.state)
        result = await call_next(tool, args)
        assert ctx.state == before, "state rebound to another middleware after call_next"
        return result

    async def after_run(self, ctx, result):
        self.seen.append(ctx.state["n"])
        return result


def test_each_middleware_has_its_own_state() -> None:
    a, b = Counter("a"), Counter("b")
    llm = MockLLMClient(responses=[_call("echo", "c1", text="x"), _answer()])
    Agent(name="b", llm=llm, tools=[FunctionTool(name="echo", fn=echo)], middleware=[a, b]).run(
        "go"
    )
    assert a.seen == [2] and b.seen == [2]


def test_state_is_saved_in_checkpoints_only_when_there_is_some(tmp_path) -> None:
    store = _store(tmp_path)
    llm = MockLLMClient(responses=[_call("echo", "c1", text="x"), _answer()])
    Agent(
        name="b",
        llm=llm,
        tools=[FunctionTool(name="echo", fn=echo)],
        middleware=[Counter("counter")],
        checkpointer=store,
    ).run("go", execution_id="s1")
    snapshots = [c.state_snapshot for c in store.list("s1")]
    assert snapshots[-1]["middleware_state"] == {"counter": {"n": 2}}

    plain = _store(tmp_path, "plain.db")
    Agent(
        name="b",
        llm=MockLLMClient(responses=[_call("echo", "c1", text="x"), _answer()]),
        tools=[FunctionTool(name="echo", fn=echo)],
        middleware=[TrimLongMessages()],
        checkpointer=plain,
    ).run("go", execution_id="s2")
    assert all("middleware_state" not in c.state_snapshot for c in plain.list("s2"))


def test_state_that_cannot_be_saved_fails_the_run_loudly(tmp_path) -> None:
    class Bad(AgentMiddleware):
        name = "bad"

        async def before_model(self, ctx, messages):
            ctx.state["when"] = object()
            return messages

    agent = Agent(
        name="b",
        llm=MockLLMClient(responses=[_call("echo", "c1", text="x"), _answer()]),
        tools=[FunctionTool(name="echo", fn=echo)],
        middleware=[Bad()],
        checkpointer=_store(tmp_path),
    )
    with pytest.raises(TypeError, match="bad"):
        agent.run("go")
    # Without a checkpointer nothing is saved, so nothing has to be JSON.
    Agent(
        name="b",
        llm=MockLLMClient(responses=[_call("echo", "c1", text="x"), _answer()]),
        tools=[FunctionTool(name="echo", fn=echo)],
        middleware=[Bad()],
    ).run("go")


# ── ToolBudget across resume (B1-5) ────────────────────────────────────────


def _budget_agent(store) -> Agent:
    llm = MockLLMClient(
        responses=[
            _call("echo", "c1", text="one"),  # call 1
            _call("approve", "c2", text="two"),  # call 2 — pauses
            _call("echo", "c3", text="three"),  # call 3 — over a budget of 2
            _answer("should not get here"),
        ]
    )
    return Agent(
        name="budget",
        llm=llm,
        tools=[FunctionTool(name="echo", fn=echo), FunctionTool(name="approve", fn=_approve)],
        middleware=[ToolBudget(max_calls=2)],
        checkpointer=store,
    )


@pytest.mark.parametrize("mode", ["arun", "astream"])
@pytest.mark.asyncio
async def test_tool_budget_counts_across_a_resume(mode: str, tmp_path) -> None:
    agent = _budget_agent(_store(tmp_path))
    if mode == "arun":
        assert (await agent.arun("go", execution_id="tb")).status == "paused"
    else:
        [e async for e in agent.astream("go", execution_id="tb")]
    resumed = await agent.aresume("tb", resume_value=Resume(approved=True))
    assert resumed.stop_reason == "tool_budget"
    assert resumed.output == "Tool budget exhausted."


@pytest.mark.asyncio
async def test_a_fork_starts_from_the_saved_state(tmp_path) -> None:
    store = _store(tmp_path)
    seen: list[dict[str, Any]] = []

    class Peek(AgentMiddleware):
        name = "peek"

        async def before_model(self, ctx, messages):
            ctx.state["turns"] = ctx.state.get("turns", 0) + 1
            seen.append(dict(ctx.state))
            return messages

    llm = MockLLMClient(responses=[_call("echo", "c1", text="x"), _answer(), _answer("forked")])
    agent = Agent(
        name="f",
        llm=llm,
        tools=[FunctionTool(name="echo", fn=echo)],
        middleware=[Peek()],
        checkpointer=store,
    )
    await agent.arun("go", execution_id="fk")
    turn1 = next(c for c in store.list("fk") if c.node_id == "turn:1")
    assert json.loads(json.dumps(turn1.state_snapshot["middleware_state"])) == {
        "peek": {"turns": 1}
    }
    seen.clear()
    await agent.afork("fk", checkpoint_id=turn1.checkpoint_id)
    assert seen[0] == {"turns": 2}
