"""Run limits and cancel (1.91.0, 2I).

``AgentConfig(timeout_seconds=, max_total_tokens=, max_cost_usd=)`` and
``agent.cancel(execution_id)``:

* ``timeout_seconds`` — the run's wall clock. ``arun`` / ``run`` / ``aresume``
  stop the moment it passes, even inside a model or tool call, and raise
  ``AgentTimeoutError`` (declared since early releases, raised by nothing until
  now). ``astream`` checks at each event.
* ``max_total_tokens`` / ``max_cost_usd`` — checked before every model call on the
  run's running total (the same one ``AgentResult.cost`` and ``cost_limit`` read).
  Over budget, the run ends cooperatively: ``stop_reason`` names the cap, no
  exception (decision B2-7). A run whose spend cannot be priced cannot be held to
  ``max_cost_usd`` and says so (``RunLimitError``) rather than passing.
* ``agent.cancel(execution_id)`` — stops a run in progress in this process.
  ``arun`` returns ``status="cancelled"``; ``astream`` ends with
  ``StreamDone(finish_reason="cancelled")``. A cancelled or timed-out run's
  checkpoint ends ``failed`` with ``run_reason`` ``"cancelled"`` / ``"timeout"``
  (decision B2-8: no new checkpoint status for the plane).
* The limits are part of ``Agent.to_dict()`` only when set (signed off
  2026-10-19), so a default agent's payload is unchanged.

Real ``Agent`` + ``FunctionModel`` (scripted model) and real checkpointers; the
cost cap uses a real ``LLMClient`` against ``tests.conftest.CaptureServer``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from fastaiagent import Agent, AgentConfig, FunctionTool, LLMClient, SQLiteCheckpointer
from fastaiagent._internal.errors import AgentTimeoutError, RunLimitError
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import ToolCall
from fastaiagent.llm.stream import StreamDone, TextDelta
from fastaiagent.testing import FunctionModel


def _calls_forever(usage: tuple[int, int] = (100, 50)) -> FunctionModel:
    """A model that asks for the ``step`` tool on every turn and reports usage."""
    count = {"n": 0}

    def respond(messages):
        count["n"] += 1
        return LLMResponse(
            content="",
            tool_calls=[ToolCall(id=f"c{count['n']}", name="step", arguments={})],
            usage={
                "prompt_tokens": usage[0],
                "completion_tokens": usage[1],
                "total_tokens": sum(usage),
            },
            finish_reason="tool_calls",
        )

    model = FunctionModel(respond)
    model.count = count  # type: ignore[attr-defined]
    return model


def step() -> str:
    """One step."""
    return "ok"


def slow(seconds: float = 3.0) -> str:
    """Take a while."""
    time.sleep(seconds)
    return "done"


async def aslow() -> str:
    """Take a while (async)."""
    await asyncio.sleep(5)
    return "done"


def _slow_agent(tool_fn: Any = aslow, **config: Any) -> tuple[Agent, Any]:
    calls = {"n": 0}

    def respond(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return "", [{"name": tool_fn.__name__, "arguments": {}}]
        return "finished"

    agent = Agent(
        name="slowpoke",
        llm=FunctionModel(respond),
        tools=[FunctionTool(name=tool_fn.__name__, fn=tool_fn)],
        config=AgentConfig(**config),
    )
    return agent, calls


def _run_end(store: SQLiteCheckpointer, execution_id: str) -> Any:
    last = store.get_last(execution_id)
    assert last is not None and last.node_id == "run_end", last
    return last


# ── max_total_tokens ───────────────────────────────────────────────────────


def test_a_token_cap_stops_the_run_before_the_next_model_call() -> None:
    model = _calls_forever()
    agent = Agent(
        name="capped",
        llm=model,
        tools=[FunctionTool(name="step", fn=step)],
        config=AgentConfig(max_iterations=20, max_total_tokens=400),
    )
    result = agent.run("go")
    assert result.status == "completed"
    assert result.stop_reason == "max_total_tokens"
    # 150 tokens a call: the 3rd call reaches 450 > 400, so no 4th call is made.
    assert model.count["n"] == 3
    assert result.tokens_used == 450


def test_a_token_cap_stops_a_streamed_run() -> None:
    model = _calls_forever()
    agent = Agent(
        name="capped-stream",
        llm=model,
        tools=[FunctionTool(name="step", fn=step)],
        config=AgentConfig(max_iterations=20, max_total_tokens=400),
    )
    result = agent.stream("go")
    assert result.stop_reason == "max_total_tokens"
    assert model.count["n"] == 3


def test_an_unset_cap_changes_nothing() -> None:
    model = _calls_forever()
    agent = Agent(
        name="uncapped",
        llm=model,
        tools=[FunctionTool(name="step", fn=step)],
        config=AgentConfig(max_iterations=4),
    )
    from fastaiagent._internal.errors import MaxIterationsError

    with pytest.raises(MaxIterationsError):
        agent.run("go")
    assert model.count["n"] == 4, "no cap: the loop ran to max_iterations, as before"


def test_limits_must_be_positive() -> None:
    for field in ("timeout_seconds", "max_total_tokens", "max_cost_usd"):
        with pytest.raises(ValueError):
            AgentConfig(**{field: 0})


# ── max_cost_usd ───────────────────────────────────────────────────────────

OPENAI_CALL = {
    "id": "c1",
    "model": "gpt-5-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "step", "arguments": "{}"},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    # gpt-5-mini: 1000 x $0.25/M + 1000 x $2/M = $0.00225 a call.
    "usage": {"prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000},
}


def test_a_cost_cap_stops_the_run(capture_server) -> None:
    capture_server.set_response_sequence([(200, OPENAI_CALL)] * 10)
    agent = Agent(
        name="cost-capped",
        llm=LLMClient(
            provider="openai", model="gpt-5-mini", api_key="k", base_url=capture_server.url
        ),
        tools=[FunctionTool(name="step", fn=step)],
        config=AgentConfig(max_iterations=10, max_cost_usd=0.005),
    )
    result = agent.run("go", trace=False)
    assert result.stop_reason == "max_cost_usd"
    # $0.00225, $0.0045, $0.00675 > $0.005 → three calls, no fourth.
    assert len(capture_server.requests) == 3
    assert result.cost == pytest.approx(0.00675)


def test_a_cost_cap_on_an_unpriced_model_says_it_cannot_check(capture_server) -> None:
    capture_server.set_response_sequence([(200, {**OPENAI_CALL, "model": "my-fine-tune"})] * 5)
    agent = Agent(
        name="unpriced",
        # A private fine-tune: the table has no rate for it.
        llm=LLMClient(
            provider="openai", model="my-fine-tune", api_key="k", base_url=capture_server.url
        ),
        tools=[FunctionTool(name="step", fn=step)],
        config=AgentConfig(max_iterations=5, max_cost_usd=1.0),
    )
    with pytest.raises(RunLimitError, match="max_cost_usd"):
        agent.run("go", trace=False)
    assert len(capture_server.requests) == 1, "refused at the first check after an unpriced call"


# ── timeout_seconds ────────────────────────────────────────────────────────


def test_a_timeout_stops_a_run_inside_a_slow_tool(tmp_path) -> None:
    store = SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))
    agent, _ = _slow_agent(timeout_seconds=0.3)
    agent._checkpointer = store
    started = time.monotonic()
    with pytest.raises(AgentTimeoutError, match="0.3"):
        agent.run("go", execution_id="to-1")
    assert time.monotonic() - started < 3, "the tool's 5 s sleep was not interrupted"
    end = _run_end(store, "to-1")
    assert end.status == "failed"
    assert end.state_snapshot["run_reason"] == "timeout"
    assert "AgentTimeoutError" in end.state_snapshot["run_error"]


def test_a_timeout_stops_a_streamed_run_at_its_next_event(tmp_path) -> None:
    store = SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))
    agent, _ = _slow_agent(tool_fn=slow, timeout_seconds=0.2)
    agent._checkpointer = store

    async def drain() -> None:
        async for _ in agent.astream("go", execution_id="to-2"):
            pass

    with pytest.raises(AgentTimeoutError):
        asyncio.run(drain())
    end = _run_end(store, "to-2")
    assert (end.status, end.state_snapshot["run_reason"]) == ("failed", "timeout")


def test_a_run_within_its_timeout_is_untouched() -> None:
    agent, _ = _slow_agent(tool_fn=step, timeout_seconds=30)
    assert agent.run("go").output == "finished"


# ── cancel ─────────────────────────────────────────────────────────────────


def test_cancel_stops_a_running_arun(tmp_path) -> None:
    store = SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))
    agent, _ = _slow_agent()
    agent._checkpointer = store

    async def main() -> tuple[Any, bool, float]:
        task = asyncio.ensure_future(agent.arun("go", execution_id="cx-1"))
        await asyncio.sleep(0.3)
        started = time.monotonic()
        found = agent.cancel("cx-1")
        result = await task
        return result, found, time.monotonic() - started

    result, found, waited = asyncio.run(main())
    assert found is True
    assert waited < 1.5
    assert result.status == "cancelled"
    assert result.stop_reason == "cancelled"
    assert result.execution_id == "cx-1"
    end = _run_end(store, "cx-1")
    assert (end.status, end.state_snapshot["run_reason"]) == ("failed", "cancelled")


def test_cancel_from_another_thread_stops_a_sync_run() -> None:
    agent, _ = _slow_agent()
    out: dict[str, Any] = {}
    worker = threading.Thread(target=lambda: out.update(r=agent.run("go", execution_id="cx-2")))
    worker.start()
    time.sleep(0.3)
    assert agent.cancel("cx-2") is True
    worker.join(timeout=3)
    assert not worker.is_alive()
    assert out["r"].status == "cancelled"


def test_cancel_ends_a_stream_with_a_cancelled_marker() -> None:
    model = _calls_forever()
    agent = Agent(
        name="streamer",
        llm=model,
        tools=[FunctionTool(name="step", fn=step)],
        config=AgentConfig(max_iterations=50),
    )

    async def main() -> list[Any]:
        events = []
        async for event in agent.astream("go", execution_id="cx-3"):
            events.append(event)
            if model.count["n"] == 2:
                agent.cancel("cx-3")
        return events

    events = asyncio.run(main())
    assert isinstance(events[-1], StreamDone)
    assert events[-1].finish_reason == "cancelled"
    assert model.count["n"] <= 3


def test_stream_returns_a_cancelled_result() -> None:
    model = _calls_forever()
    agent = Agent(
        name="stream-sync",
        llm=model,
        tools=[FunctionTool(name="step", fn=lambda: (agent.cancel(agent_exec["id"]), "ok")[1])],
        config=AgentConfig(max_iterations=50),
    )
    agent_exec = {"id": "cx-4"}
    result = agent.stream("go", execution_id="cx-4")
    assert result.status == "cancelled"


def test_cancel_of_an_unknown_run_is_false() -> None:
    agent, _ = _slow_agent()
    assert agent.cancel("no-such-run") is False


def test_a_finished_run_cannot_be_cancelled() -> None:
    agent, _ = _slow_agent(tool_fn=step)
    agent.run("go", execution_id="cx-5")
    assert agent.cancel("cx-5") is False


def test_cancelling_the_callers_task_still_cancels_the_run() -> None:
    """An outside cancel (the caller's own task) is not a ``cancel()``: it
    propagates as ``CancelledError`` and the run does not keep going."""
    agent, calls = _slow_agent()

    async def main() -> None:
        task = asyncio.ensure_future(agent.arun("go", execution_id="cx-6"))
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.2)

    asyncio.run(main())
    assert calls["n"] == 1, "the run kept calling the model after its caller was cancelled"


# ── serialization (signed off 2026-10-19: only when set) ───────────────────


def test_limits_are_in_to_dict_only_when_set() -> None:
    default = Agent(name="d", llm=FunctionModel(lambda m: "x")).to_dict()["config"]
    for field in ("timeout_seconds", "max_total_tokens", "max_cost_usd"):
        assert field not in default
    limited = Agent(
        name="l",
        llm=LLMClient(provider="openai", model="gpt-5-mini"),
        config=AgentConfig(timeout_seconds=30, max_total_tokens=5000, max_cost_usd=0.5),
    )
    data = limited.to_dict()["config"]
    assert (data["timeout_seconds"], data["max_total_tokens"], data["max_cost_usd"]) == (
        30,
        5000,
        0.5,
    )
    back = Agent.from_dict(limited.to_dict())
    assert back.config.max_cost_usd == 0.5 and back.config.timeout_seconds == 30


def test_a_text_delta_is_still_streamed_under_a_cap() -> None:
    """A cap that never trips leaves the stream as it was."""
    agent = Agent(
        name="hello",
        llm=FunctionModel(lambda m: "hello there"),
        config=AgentConfig(max_total_tokens=10_000, timeout_seconds=30),
    )

    async def drain() -> str:
        return "".join([e.text async for e in agent.astream("hi") if isinstance(e, TextDelta)])

    assert asyncio.run(drain()) == "hello there"


# ── the sweep: resume, containers, nested runs ─────────────────────────────


def test_a_resumed_run_is_held_to_its_timeout(tmp_path) -> None:
    from fastaiagent import Resume
    from fastaiagent.chain.interrupt import interrupt

    async def approve() -> str:
        """Needs approval, then works slowly."""
        interrupt(reason="approve?", context={})
        await asyncio.sleep(5)
        return "approved"

    calls = {"n": 0}

    def respond(messages):
        calls["n"] += 1
        return ("", [{"name": "approve", "arguments": {}}]) if calls["n"] == 1 else "done"

    store = SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))
    agent = Agent(
        name="resumable",
        llm=FunctionModel(respond),
        tools=[FunctionTool(name="approve", fn=approve)],
        checkpointer=store,
        config=AgentConfig(timeout_seconds=0.3),
    )
    assert agent.run("go", execution_id="rs-1").status == "paused"
    started = time.monotonic()
    with pytest.raises(AgentTimeoutError):
        agent.resume("rs-1", resume_value=Resume(approved=True))
    assert time.monotonic() - started < 3
    assert _run_end(store, "rs-1").state_snapshot["run_reason"] == "timeout"


def test_a_supervisor_run_can_be_cancelled_by_its_id() -> None:
    from fastaiagent import Supervisor, Worker

    worker = Agent(
        name="w",
        llm=FunctionModel(lambda m: ("", [{"name": "aslow", "arguments": {}}])),
        tools=[FunctionTool(name="aslow", fn=aslow)],
    )
    lead_turns = {"n": 0}

    def lead(messages):
        lead_turns["n"] += 1
        return (
            ("", [{"name": "delegate_to_w", "arguments": {"task": "go"}}])
            if lead_turns["n"] == 1
            else "x"
        )

    sup = Supervisor(name="lead", llm=FunctionModel(lead), workers=[Worker(agent=worker)])

    async def main() -> Any:
        task = asyncio.ensure_future(sup.arun("go", execution_id="sup-1"))
        await asyncio.sleep(0.3)
        assert Agent.cancel(worker, "sup-1") is True
        return await asyncio.wait_for(task, 3)

    assert asyncio.run(main()).status == "cancelled"


def test_a_swarm_is_not_cancellable_and_says_so() -> None:
    """Its hops run nested; 1.91.0 registers no Swarm-level run (documented)."""
    from fastaiagent import Swarm

    agent = Agent(name="a", llm=FunctionModel(lambda m: "done"))
    swarm = Swarm(name="s", agents=[agent], entrypoint="a", handoffs={"a": []})
    assert swarm.run("go", execution_id="sw-1").output == "done"
    assert agent.cancel("sw-1") is False


def test_a_nested_agent_holds_its_own_caps() -> None:
    """An agent called from a tool keeps its own token cap, on its own spend."""
    inner_model = _calls_forever()
    inner = Agent(
        name="inner",
        llm=inner_model,
        tools=[FunctionTool(name="step", fn=step)],
        config=AgentConfig(max_iterations=20, max_total_tokens=300),
    )

    async def ask_inner() -> str:
        """Ask the inner agent."""
        return (await inner.arun("go")).stop_reason or ""

    outer_turns = {"n": 0}

    def outer(messages):
        outer_turns["n"] += 1
        return ("", [{"name": "ask_inner", "arguments": {}}]) if outer_turns["n"] == 1 else "ok"

    agent = Agent(
        name="outer",
        llm=FunctionModel(outer),
        tools=[FunctionTool(name="ask_inner", fn=ask_inner)],
    )
    assert agent.run("go").output == "ok"
    assert inner_model.count["n"] == 2, "300 tokens: two 150-token calls, then stopped"


def test_a_nested_agent_stops_with_the_top_level_run(tmp_path) -> None:
    inner = Agent(
        name="inner",
        llm=FunctionModel(lambda m: ("", [{"name": "aslow", "arguments": {}}])),
        tools=[FunctionTool(name="aslow", fn=aslow)],
    )

    async def ask_inner() -> str:
        """Ask the inner agent."""
        return (await inner.arun("go")).output

    agent = Agent(
        name="outer",
        llm=FunctionModel(lambda m: ("", [{"name": "ask_inner", "arguments": {}}])),
        tools=[FunctionTool(name="ask_inner", fn=ask_inner)],
        config=AgentConfig(timeout_seconds=0.3),
    )
    started = time.monotonic()
    with pytest.raises(AgentTimeoutError):
        agent.run("go")
    assert time.monotonic() - started < 3


@pytest.mark.parametrize("trace", [True, False], ids=["traced", "untraced"])
def test_cancel_works_with_tracing_on_and_off(trace: bool) -> None:
    agent, _ = _slow_agent()

    async def main() -> Any:
        task = asyncio.ensure_future(agent.arun("go", execution_id=f"cx-t{trace}", trace=trace))
        await asyncio.sleep(0.3)
        agent.cancel(f"cx-t{trace}")
        return await task

    assert asyncio.run(main()).status == "cancelled"
