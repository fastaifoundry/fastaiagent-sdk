"""Five defects found while mapping Bets 1 + 2, each reproduced by running it (1.91.0).

* **X1** — ``Agent.astream`` wrote no run-end marker, so a FINISHED streamed run
  looked crashed and ``aresume`` re-ran it: the model was called again and every
  tool fired again (the audit-D5 class, on the stream path).
* **X2** — when the call ``aresume`` re-invokes fails (a ``tool_call`` guardrail
  blocks it), the run kept ``interrupted`` as its newest row with the pending row
  already claimed: it looked paused and could never be resumed, and nothing said
  it had failed.
* **X3** — ``Agent.astream`` opened no run-cost scope, so ``cost_limit`` could not
  read the spend and blocked every streamed run.
* **X8** — the ``LLMResponse`` a streamed turn hands ``after_model`` had no
  ``total_tokens``, so a token-budget middleware (the documented ``TokenCounter``)
  counted nothing on ``astream``.
* **X10** — ``PlatformKB.as_tool()`` reported ``origin="function"`` where
  ``LocalKB`` says ``"kb"``, so the Local UI listed a knowledge base as a tool.

Real ``Agent``, real ``SQLiteCheckpointer``, real guardrails; the model is the
repo's ``MockLLMClient`` (a real ``LLMClient`` subclass) or ``FunctionModel``.
"""

from __future__ import annotations

from typing import Any

import pytest

from fastaiagent import Agent, FunctionTool, SQLiteCheckpointer
from fastaiagent.agent.middleware import AgentMiddleware
from fastaiagent.chain.checkpoint import is_run_end
from fastaiagent.chain.interrupt import AlreadyResumed, Resume, interrupt
from fastaiagent.guardrail import Guardrail, GuardrailPosition, GuardrailResult
from fastaiagent.guardrail.builtins import cost_limit
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import ToolCall
from tests.conftest import MockLLMClient

fired: list[int] = []


def charge(amount: int) -> str:
    """A side effect: one call is one charge."""
    fired.append(amount)
    return f"charged {amount}"


def _tool_then_answer() -> MockLLMClient:
    return MockLLMClient(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name="charge", arguments={"amount": 5})],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
            LLMResponse(
                content="done",
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
            ),
        ]
    )


def _store(tmp_path, name: str = "cp.db") -> SQLiteCheckpointer:
    store = SQLiteCheckpointer(db_path=str(tmp_path / name))
    store.setup()
    return store


async def _drain(agent: Agent, *args: Any, **kwargs: Any) -> list[Any]:
    return [event async for event in agent.astream(*args, **kwargs)]


# ── X1: a finished streamed run is finished ────────────────────────────────


@pytest.mark.asyncio
async def test_a_finished_stream_writes_one_completed_marker(tmp_path) -> None:
    store = _store(tmp_path)
    agent = Agent(
        name="x1",
        llm=_tool_then_answer(),
        tools=[FunctionTool(name="charge", fn=charge)],
        checkpointer=store,
    )
    await _drain(agent, "go", execution_id="x1-run")

    markers = [c for c in store.list("x1-run") if is_run_end(c)]
    assert [m.status for m in markers] == ["completed"]
    assert is_run_end(store.get_last("x1-run"))


@pytest.mark.asyncio
async def test_resuming_a_finished_stream_is_refused_not_replayed(tmp_path) -> None:
    """THE headline: before 1.91.0 this called the model again and re-charged."""
    store = _store(tmp_path)
    fired.clear()
    agent = Agent(
        name="x1",
        llm=_tool_then_answer(),
        tools=[FunctionTool(name="charge", fn=charge)],
        checkpointer=store,
    )
    await _drain(agent, "go", execution_id="x1-done")
    assert fired == [5]
    calls_before = agent.llm._call_count

    with pytest.raises(AlreadyResumed) as excinfo:
        await agent.aresume("x1-done")
    assert "already finished" in str(excinfo.value)
    assert fired == [5]
    assert agent.llm._call_count == calls_before


@pytest.mark.asyncio
async def test_a_stream_that_fails_is_marked_failed(tmp_path) -> None:
    store = _store(tmp_path)

    class Down(MockLLMClient):
        async def astream(self, messages, tools=None, **kwargs):
            raise RuntimeError("provider down")
            yield  # pragma: no cover - makes this an async generator

    agent = Agent(name="x1", llm=Down(), checkpointer=store)
    with pytest.raises(RuntimeError):
        await _drain(agent, "go", execution_id="x1-fail")
    last = store.get_last("x1-fail")
    assert is_run_end(last) and last.status == "failed"
    assert "provider down" in last.state_snapshot["run_error"]


@pytest.mark.parametrize("mode", ["arun", "astream"])
@pytest.mark.asyncio
async def test_a_failure_after_the_loop_does_not_refire_the_tool_on_resume(
    mode: str, tmp_path
) -> None:
    """The ``arun`` rule, now on the stream path too: a run whose loop ended on a
    tool turn (here ``wrap_tool`` ran the tool and then stopped the run) and then
    failed (an output guardrail blocks) leaves a re-entry point past that turn,
    so crash recovery re-asks the model instead of re-running the tool."""
    from fastaiagent._internal.errors import StopAgent

    class RunThenStop(AgentMiddleware):
        async def wrap_tool(self, ctx, tool, args, call_next):
            await call_next(tool, args)
            raise StopAgent("enough", reason="budget")

    store = _store(tmp_path)
    fired.clear()
    agent = Agent(
        name="x1",
        llm=_tool_then_answer(),
        tools=[FunctionTool(name="charge", fn=charge)],
        guardrails=[
            Guardrail(
                name="block-all",
                position=GuardrailPosition.output,
                fn=lambda text: GuardrailResult(passed=False, message="blocked"),
            )
        ],
        middleware=[RunThenStop()],
        checkpointer=store,
    )
    with pytest.raises(Exception, match="blocked"):
        if mode == "arun":
            await agent.arun("go", execution_id="after-loop")
        else:
            await _drain(agent, "go", execution_id="after-loop")
    assert fired == [5]
    assert store.get_last("after-loop").status == "failed"

    # The repaired deployment: same store, same model, no blocking rule.
    resumer = Agent(
        name="x1", llm=agent.llm, tools=[FunctionTool(name="charge", fn=charge)], checkpointer=store
    )
    await resumer.aresume("after-loop")
    assert fired == [5], "crash recovery re-ran the tool that already ran"


# ── X12: running out of iterations right after a tool turn ─────────────────


@pytest.mark.parametrize("mode", ["arun", "astream"])
@pytest.mark.asyncio
async def test_resuming_after_max_iterations_does_not_refire_the_last_tool(
    mode: str, tmp_path
) -> None:
    """The last turn's tool ran, then the loop ran out of iterations. Its newest
    row was that turn's pre-tool checkpoint, so recovery re-ran the tool — one
    charge became two, on both paths (measured before the fix)."""
    from fastaiagent._internal.errors import MaxIterationsError

    store = _store(tmp_path)
    fired.clear()
    agent = Agent(
        name="x12",
        llm=_tool_then_answer(),
        tools=[FunctionTool(name="charge", fn=charge)],
        checkpointer=store,
    )
    agent.config.max_iterations = 1
    with pytest.raises(MaxIterationsError):
        if mode == "arun":
            await agent.arun("go", execution_id="x12-run")
        else:
            await _drain(agent, "go", execution_id="x12-run")
    assert fired == [5]

    agent.config.max_iterations = 5
    result = await agent.aresume("x12-run")
    assert result.output == "done"
    assert fired == [5], "crash recovery re-ran the tool that already ran"


@pytest.mark.asyncio
async def test_stopping_at_paused_never_marks_the_run(tmp_path) -> None:
    """A consumer that stops reading at ``Paused`` closes the generator; that must
    not write a marker after the ``interrupted`` row (it would hide the pause)."""
    from fastaiagent.llm.stream import Paused

    def approve(amount: int) -> str:
        """Needs approval."""
        interrupt(reason="approve?", context={"amount": amount})
        return "ok"

    store = _store(tmp_path)
    llm = MockLLMClient(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name="approve", arguments={"amount": 5})],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="approved", finish_reason="stop"),
        ]
    )
    agent = Agent(
        name="x1", llm=llm, tools=[FunctionTool(name="approve", fn=approve)], checkpointer=store
    )
    async for event in agent.astream("go", execution_id="x1-pause"):
        if isinstance(event, Paused):
            break
    last = store.get_last("x1-pause")
    assert last.status == "interrupted" and not is_run_end(last)
    done = await agent.aresume("x1-pause", resume_value=Resume(approved=True))
    assert done.status == "completed"


# ── X2: a resume whose re-invoked call fails is marked failed ──────────────


@pytest.mark.asyncio
async def test_a_resume_whose_reinvoked_call_is_blocked_is_marked_failed(tmp_path) -> None:
    seen = {"n": 0}

    def second_time_blocks(text: str) -> GuardrailResult:
        seen["n"] += 1
        return GuardrailResult(passed=seen["n"] < 2, message="blocked on resume")

    def pay(amount: int) -> str:
        """Needs approval."""
        interrupt(reason="approve?", context={"amount": amount})
        return "paid"

    store = _store(tmp_path)
    llm = MockLLMClient(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name="pay", arguments={"amount": 5})],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="paid", finish_reason="stop"),
        ]
    )
    agent = Agent(
        name="x2",
        llm=llm,
        tools=[FunctionTool(name="pay", fn=pay)],
        guardrails=[
            Guardrail(name="g", position=GuardrailPosition.tool_call, fn=second_time_blocks)
        ],
        checkpointer=store,
    )
    assert (await agent.arun("pay", execution_id="x2-run")).status == "paused"

    with pytest.raises(Exception, match="blocked on resume"):
        await agent.aresume("x2-run", resume_value=Resume(approved=True))

    last = store.get_last("x2-run")
    assert is_run_end(last) and last.status == "failed"
    assert "blocked on resume" in last.state_snapshot["run_error"]


# ── X3: cost_limit works on a streamed run ─────────────────────────────────


@pytest.mark.asyncio
async def test_cost_limit_gives_a_verdict_on_astream() -> None:
    agent = Agent(
        name="x3",
        llm=_tool_then_answer(),
        tools=[FunctionTool(name="charge", fn=charge)],
        guardrails=[cost_limit(1.0, GuardrailPosition.output)],
    )
    events = await _drain(agent, "go")
    assert type(events[-1]).__name__ == "StreamDone"


def test_stream_still_reports_the_runs_tokens_and_cost() -> None:
    """``stream()`` already opened a scope; ``astream`` must share it, not shadow
    it with an empty one."""
    agent = Agent(
        name="x3", llm=_tool_then_answer(), tools=[FunctionTool(name="charge", fn=charge)]
    )
    result = agent.stream("go")
    assert result.output == "done"
    assert result.tokens_used == 15 + 27
    assert result.cost_known is True  # MockLLMClient is unpriced but zero calls are recorded


# ── X8: streamed after_model sees total_tokens ─────────────────────────────


@pytest.mark.parametrize("mode", ["arun", "astream"])
@pytest.mark.asyncio
async def test_after_model_sees_total_tokens_on_both_paths(mode: str) -> None:
    seen: list[Any] = []

    class Count(AgentMiddleware):
        async def after_model(self, ctx, response):
            seen.append(response.usage.get("total_tokens"))
            return response

    agent = Agent(
        name="x8",
        llm=_tool_then_answer(),
        tools=[FunctionTool(name="charge", fn=charge)],
        middleware=[Count()],
    )
    if mode == "arun":
        await agent.arun("go")
    else:
        await _drain(agent, "go")
    assert seen == [15, 27]


# ── X10: a platform KB tool is a KB ────────────────────────────────────────


def test_platform_kb_tool_is_marked_as_a_kb() -> None:
    from fastaiagent.kb.platform import PlatformKB

    platform_tool = PlatformKB(kb_id="docs").as_tool()
    assert platform_tool.name == "search_docs"
    assert platform_tool.origin == "kb"
    assert platform_tool.to_dict()["origin"] == "kb"
