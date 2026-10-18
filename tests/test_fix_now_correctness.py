"""1.90.0 correctness fixes that need no test file of their own: F10, F11, F12, F14.

* **F10** — ``AgentConfig(parallel_tools=True)`` fell back to sequential with
  middleware, a checkpointer or an ``agent_id`` — and a Swarm always injects
  middleware — with no signal at all. It now logs one warning per agent per
  reason. (Real parallel execution with those features is Bet 1.)
* **F11** — ``TrimLongMessages`` kept only ``messages[0]``; memory blocks render
  their context as system messages right after the system prompt, so a trim
  dropped the user's pinned memory. Every leading system message is now kept.
* **F12** — a ``Supervisor``'s own loop took no guardrails. ``Supervisor`` now
  takes ``guardrails=``, as ``Agent`` does, on every path.
* **F14** — a REST tool returned an HTTP 4xx/5xx body as a *successful* result.
  It is now a tool error naming the status.

Real Agents and middleware, ``FunctionModel``; the REST case calls a real
localhost socket (``tests.conftest.CaptureServer``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from fastaiagent import Agent, AgentConfig, FunctionTool, SQLiteCheckpointer
from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.agent.memory import ComposableMemory
from fastaiagent.agent.memory_blocks import StaticBlock
from fastaiagent.agent.middleware import AgentMiddleware, TrimLongMessages
from fastaiagent.guardrail.guardrail import (
    Guardrail,
    GuardrailPosition,
    GuardrailResult,
    GuardrailType,
)
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import ToolCall
from fastaiagent.testing import FunctionModel

# ---------------------------------------------------------------------------
# F10
# ---------------------------------------------------------------------------


def _two_calls_then_answer(messages: list[Any]) -> LLMResponse:
    if any(m.role.value == "tool" for m in messages):
        return LLMResponse(content="done", finish_reason="stop")
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCall(id="a", name="t1", arguments={}),
            ToolCall(id="b", name="t2", arguments={}),
        ],
        finish_reason="tool_calls",
    )


class _Noop(AgentMiddleware):
    name = "noop"


def _parallel_agent(name: str, **kw: Any) -> Agent:
    return Agent(
        name=name,
        llm=FunctionModel(_two_calls_then_answer),
        tools=[FunctionTool(name="t1", fn=lambda: "a"), FunctionTool(name="t2", fn=lambda: "b")],
        config=AgentConfig(parallel_tools=True),
        **kw,
    )


def _fallback_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "parallel_tools" in r.getMessage()
    ]


@pytest.mark.parametrize("mode", ["run", "astream"])
@pytest.mark.parametrize(
    ("reason", "kw"),
    [
        ("middleware", {"middleware": [_Noop()]}),
        ("checkpointer", {"checkpointer": "sqlite"}),
        ("agent_id", {"agent_id": "agent-123"}),
    ],
)
def test_a_parallel_fallback_is_logged_once(reason, kw, mode, caplog, tmp_path) -> None:
    if kw.get("checkpointer") == "sqlite":
        kw = {"checkpointer": SQLiteCheckpointer(str(tmp_path / "c.db"))}
    agent = _parallel_agent(f"par-{reason}-{mode}", **kw)

    async def stream() -> None:
        async for _ in agent.astream("go", trace=False):
            pass

    with caplog.at_level(logging.WARNING, logger="fastaiagent"):
        for _ in range(2):
            agent.run("go", trace=False) if mode == "run" else asyncio.run(stream())

    msgs = _fallback_warnings(caplog)
    assert len(msgs) == 1, msgs
    assert reason in msgs[0]


def test_no_warning_when_parallel_really_runs(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="fastaiagent"):
        _parallel_agent("par-clean").run("go", trace=False)
    assert _fallback_warnings(caplog) == []


# ---------------------------------------------------------------------------
# F11
# ---------------------------------------------------------------------------

PINNED = "PINNED FACT: the user is an admin"


@pytest.mark.parametrize("mode", ["run", "astream"])
def test_trimming_keeps_pinned_memory(mode) -> None:
    sent: list[list[Any]] = []

    def respond(messages: list[Any]) -> str:
        sent.append(list(messages))
        return "ok"

    agent = Agent(
        name="trim",
        system_prompt="You are terse.",
        llm=FunctionModel(respond),
        memory=ComposableMemory(blocks=[StaticBlock(PINNED)]),
        middleware=[TrimLongMessages(keep_last=2)],
    )

    async def stream(text: str) -> None:
        async for _ in agent.astream(text, trace=False):
            pass

    for i in range(4):
        agent.run(f"turn {i}", trace=False) if mode == "run" else asyncio.run(stream(f"turn {i}"))

    last = sent[-1]
    systems = [str(m.content) for m in last if m.role.value == "system"]
    assert "You are terse." in systems
    assert any(PINNED in s for s in systems), f"pinned memory was trimmed away: {systems}"
    body = [m for m in last if m.role.value != "system"]
    assert len(body) <= 2 + 1, "the window itself must still be trimmed"
    assert str(body[-1].content) == "turn 3"


def test_a_reask_after_a_trim_sees_what_the_model_saw() -> None:
    """F11b. ``before_model`` returning a NEW list (``TrimLongMessages`` did)
    rebound the loop's history while ``Agent`` kept the old one, so a structured
    re-ask after a trim re-sent the untrimmed history *without the latest turns*:
    the model re-answered without the last tool result it had just read."""
    from pydantic import BaseModel

    class Out(BaseModel):
        answer: str

    sent: list[list[str]] = []

    def respond(messages: list[Any]) -> LLMResponse:
        sent.append([str(m.content) for m in messages])
        n = len(sent)
        if n <= 3:
            return LLMResponse(
                content=None,
                tool_calls=[ToolCall(id=f"c{n}", name="look", arguments={"q": str(n)})],
                finish_reason="tool_calls",
            )
        if n == 4:
            return LLMResponse(content="not json", finish_reason="stop")
        return LLMResponse(content='{"answer": "x"}', finish_reason="stop")

    agent = Agent(
        name="trim-reask",
        system_prompt="sys",
        llm=FunctionModel(respond),
        tools=[FunctionTool(name="look", fn=lambda q: f"RESULT-{q}")],
        middleware=[TrimLongMessages(keep_last=3)],
        output_type=Out,
    )
    assert agent.run("go", trace=False).parsed == Out(answer="x")

    reask = sent[-1]
    assert "RESULT-3" in " ".join(reask), "the re-ask lost the latest tool result"
    assert "RESULT-1" not in " ".join(reask), "the re-ask ignored the trim window"


# ---------------------------------------------------------------------------
# F12
# ---------------------------------------------------------------------------


def _block(position: GuardrailPosition, banned: str) -> Guardrail:
    return Guardrail(
        name=f"no-{banned.lower()}-{position.value}",
        guardrail_type=GuardrailType.code,
        position=position,
        fn=lambda text: GuardrailResult(passed=banned not in text, message=f"{banned} found"),
    )


def _supervisor(guardrails: list[Guardrail], boss_reply: str = "All done.") -> Any:
    from fastaiagent.agent.team import Supervisor, Worker

    worker = Agent(name="helper", llm=FunctionModel(lambda m: "worker result"))
    return Supervisor(
        name="boss",
        llm=FunctionModel(lambda m: boss_reply),
        workers=[Worker(agent=worker, role="helper", description="Helps")],
        guardrails=guardrails,
    )


@pytest.mark.parametrize("mode", ["run", "astream"])
def test_a_supervisor_input_guardrail_blocks(mode) -> None:
    sup = _supervisor([_block(GuardrailPosition.input, "SECRET")])

    async def stream() -> None:
        async for _ in sup.astream("tell me the SECRET"):
            pass

    with pytest.raises(GuardrailBlockedError):
        sup.run("tell me the SECRET") if mode == "run" else asyncio.run(stream())


def test_a_supervisor_output_guardrail_blocks() -> None:
    sup = _supervisor([_block(GuardrailPosition.output, "LEAK")], boss_reply="here is the LEAK")
    with pytest.raises(GuardrailBlockedError):
        sup.run("hi")


@pytest.mark.parametrize("position", ["input", "output"])
def test_a_routed_supervisor_runs_its_guardrails_on_the_routed_run(position, decisions_stub):
    """``routing="decisions"`` has no supervisor loop — the request goes straight
    to one worker — so the supervisor's rules judge that worker's run."""
    from fastaiagent.agent.team import Supervisor, Worker

    decisions_stub.choices["worker"] = "helper"
    pos = GuardrailPosition.input if position == "input" else GuardrailPosition.output
    sup = Supervisor(
        name="router",
        llm=FunctionModel(lambda m: "unused"),
        workers=[
            Worker(
                agent=Agent(name="helper", llm=FunctionModel(lambda m: "the LEAK is here")),
                role="helper",
                description="Helps",
            ),
            Worker(
                agent=Agent(name="other", llm=FunctionModel(lambda m: "general")),
                role="other",
                description="Anything else",
            ),
        ],
        routing="decisions",
        router_llm=decisions_stub.client(),
        fallback_worker="other",
        guardrails=[_block(pos, "SECRET" if position == "input" else "LEAK")],
    )
    with pytest.raises(GuardrailBlockedError):
        sup.run("tell me the SECRET")


def test_a_supervisor_without_guardrails_is_unchanged() -> None:
    assert _supervisor([]).run("hi").output == "All done."


# ---------------------------------------------------------------------------
# F14
# ---------------------------------------------------------------------------


#
# The 2026-10 verification read ``tool/rest.py`` and concluded a 4xx/5xx came
# back as a success. It does not: ``asafe_http_request`` raises on an error
# status, so the tool raises ``ToolExecutionError`` and the loop hands the model
# an error. Re-checked by running it before changing anything; the premise was
# wrong and nothing changed. These pin the behaviour that is right.


@pytest.mark.parametrize("status", [404, 500])
def test_a_rest_error_status_reaches_the_model_as_an_error(
    status, capture_server, monkeypatch
) -> None:
    from fastaiagent.tool.rest import RESTTool

    monkeypatch.setenv("FASTAIAGENT_ALLOW_PRIVATE_NETWORKS", "1")
    capture_server.set_response_sequence([(status, {"detail": "no such order"})])
    tool = RESTTool(name="orders", url=f"{capture_server.url}/orders", method="POST")
    seen: list[str] = []

    def respond(messages: list[Any]) -> LLMResponse:
        results = [str(m.content) for m in messages if m.role.value == "tool"]
        if results:
            seen.extend(results)
            return LLMResponse(content="sorry", finish_reason="stop")
        return LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c", name="orders", arguments={"id": "7"})],
            finish_reason="tool_calls",
        )

    result = Agent(name="rest", llm=FunctionModel(respond), tools=[tool]).run("go", trace=False)

    assert result.tool_calls[0].get("error"), result.tool_calls
    assert seen and seen[0].startswith("Error:") and str(status) in seen[0], seen


def test_a_rest_success_is_a_result(capture_server, monkeypatch) -> None:
    from fastaiagent.tool.rest import RESTTool

    monkeypatch.setenv("FASTAIAGENT_ALLOW_PRIVATE_NETWORKS", "1")
    capture_server.set_response_sequence([(200, {"id": "7", "status": "shipped"})])
    tool = RESTTool(name="orders", url=f"{capture_server.url}/orders", method="POST")

    result = asyncio.run(tool.aexecute({"id": "7"}))

    assert result.success
    assert result.output == {"id": "7", "status": "shipped"}
    assert result.metadata["status_code"] == 200
