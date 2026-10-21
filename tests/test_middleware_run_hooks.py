"""The run-level middleware hooks and per-call context (1.91.0, Bet 1 piece 1A + 1E).

``AgentMiddleware`` could only see one model call or one tool call at a time, so
anything about the *run* — timing it, retrying or caching a model call, swapping
the model for one call, seeing why it failed, filtering what is streamed — had to
be wired into the agent loop by hand. These hooks sit beside the existing three,
follow the same ordering rules, and default to doing nothing:

* ``before_run(ctx, input)`` — once per run (``arun``, ``astream``, resume), before
  the input guardrails, so they judge the input as rewritten; ``StopAgent`` ends
  the run without a model call.
* ``after_run(ctx, result)`` — once, on success; may replace the result.
* ``wrap_model(ctx, messages, call_next)`` — an onion around EVERY model call,
  the structured-output and guardrail re-asks included.
* ``on_error(ctx, error)`` — once, when the run fails; it cannot swallow it.
* ``on_stream_event(ctx, event)`` — each event ``astream`` yields; rewrite or
  drop (``Paused`` and ``StreamDone`` are always delivered).

``MiddlewareContext`` gains ``llm`` and ``tools`` (what THIS call uses — set them to
change one call), ``execution_id`` and ``usage`` (the run's spend so far).

Real ``Agent``, real checkpointer, real guardrails; the model is the repo's
``MockLLMClient`` (a real ``LLMClient`` subclass that records every call).
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from pydantic import BaseModel

from fastaiagent import Agent, AgentMiddleware, FunctionTool, SQLiteCheckpointer, StopAgent
from fastaiagent.chain.interrupt import Resume, interrupt
from fastaiagent.guardrail import Guardrail, GuardrailPosition, GuardrailResult
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import ToolCall
from fastaiagent.llm.stream import Paused, StreamDone, TextDelta, Usage
from tests.conftest import MockLLMClient


def echo(text: str) -> str:
    """Echo the text."""
    return f"echo:{text}"


def _text(*replies: str) -> MockLLMClient:
    return MockLLMClient(responses=[LLMResponse(content=r, finish_reason="stop") for r in replies])


def _tool_then(final: str = "done", name: str = "echo") -> MockLLMClient:
    return MockLLMClient(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name=name, arguments={"text": "hi"})],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
            LLMResponse(
                content=final,
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
            ),
        ]
    )


def _agent(llm: MockLLMClient, *middleware: AgentMiddleware, **kw: Any) -> Agent:
    return Agent(
        name="hooks",
        llm=llm,
        tools=kw.pop("tools", [FunctionTool(name="echo", fn=echo)]),
        middleware=list(middleware),
        **kw,
    )


async def _stream(agent: Agent, *args: Any, **kw: Any) -> list[Any]:
    return [e async for e in agent.astream(*args, **kw)]


def _user_text(call: dict[str, Any]) -> str:
    return next(m.content for m in reversed(call["messages"]) if m.role.value == "user")


# ── before_run ─────────────────────────────────────────────────────────────


class Prefix(AgentMiddleware):
    name = "prefix"

    def __init__(self) -> None:
        self.calls = 0

    async def before_run(self, ctx, input):
        self.calls += 1
        return f"[ticket 42] {input}"


@pytest.mark.parametrize("mode", ["arun", "astream"])
@pytest.mark.asyncio
async def test_before_run_rewrites_the_input_once(mode: str) -> None:
    llm = _tool_then()
    mw = Prefix()
    agent = _agent(llm, mw)
    if mode == "arun":
        await agent.arun("refund me")
    else:
        await _stream(agent, "refund me")
    assert mw.calls == 1
    assert _user_text(llm._calls[0]) == "[ticket 42] refund me"


@pytest.mark.asyncio
async def test_input_guardrails_judge_the_rewritten_input() -> None:
    seen: list[str] = []

    def record(text: str) -> GuardrailResult:
        seen.append(text)
        return GuardrailResult(passed=True)

    agent = _agent(
        _text("ok"),
        Prefix(),
        guardrails=[Guardrail(name="rec", position=GuardrailPosition.input, fn=record)],
    )
    await agent.arun("hello")
    assert seen == ["[ticket 42] hello"]


class StopAtTheDoor(AgentMiddleware):
    async def before_run(self, ctx, input):
        raise StopAgent("We are closed.", reason="closed")


@pytest.mark.parametrize("mode", ["arun", "astream"])
@pytest.mark.asyncio
async def test_before_run_can_stop_the_run_without_a_model_call(mode: str) -> None:
    llm = _text("never")
    agent = _agent(llm, StopAtTheDoor())
    if mode == "arun":
        result = await agent.arun("hi")
        assert result.output == "We are closed."
        assert result.stop_reason == "closed"
        assert result.status == "completed"
    else:
        events = await _stream(agent, "hi")
        assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "We are closed."
        assert isinstance(events[-1], StreamDone)
    assert llm._call_count == 0


# ── after_run ──────────────────────────────────────────────────────────────


class Stamp(AgentMiddleware):
    def __init__(self, tag: str, log: list[str]) -> None:
        self.tag, self.log = tag, log

    async def after_run(self, ctx, result):
        self.log.append(self.tag)
        result.output = f"{result.output} [{self.tag}]"
        return result


def test_after_run_rewrites_the_result_in_reverse_order() -> None:
    log: list[str] = []
    result = _agent(_tool_then(), Stamp("a", log), Stamp("b", log)).run("go")
    assert log == ["b", "a"]
    assert result.output == "done [b] [a]"


def test_after_run_on_stream_shapes_what_stream_returns() -> None:
    log: list[str] = []
    result = _agent(_tool_then(), Stamp("a", log)).stream("go")
    assert log == ["a"]
    assert result.output == "done [a]"


class Count(AgentMiddleware):
    def __init__(self) -> None:
        self.before = self.after = 0
        self.errors: list[BaseException] = []

    async def before_run(self, ctx, input):
        self.before += 1
        return input

    async def after_run(self, ctx, result):
        self.after += 1
        return result

    async def on_error(self, ctx, error):
        self.errors.append(error)


def _approve(text: str) -> str:
    """Needs approval."""
    interrupt(reason="approve?", context={"text": text})
    return f"approved:{text}"


@pytest.mark.asyncio
async def test_a_paused_run_gets_no_after_run_and_its_resume_is_one_run(tmp_path) -> None:
    mw = Count()
    agent = _agent(
        _tool_then(name="approve"),
        mw,
        tools=[FunctionTool(name="approve", fn=_approve)],
        checkpointer=SQLiteCheckpointer(db_path=str(tmp_path / "cp.db")),
    )
    paused = await agent.arun("go", execution_id="h-pause")
    assert paused.status == "paused"
    assert (mw.before, mw.after) == (1, 0)

    done = await agent.aresume("h-pause", resume_value=Resume(approved=True))
    assert done.status == "completed"
    assert (mw.before, mw.after) == (2, 1)
    assert mw.errors == []


@pytest.mark.asyncio
async def test_a_resume_shares_one_context_between_the_reinvoked_tool_and_the_loop(
    tmp_path,
) -> None:
    """Before 1.91.0 the re-invoked tool saw one ``scratch`` and the continued loop
    another, so a middleware that counted across the two lost the count."""
    seen: list[int] = []

    class Tally(AgentMiddleware):
        async def wrap_tool(self, ctx, tool, args, call_next):
            ctx.scratch["n"] = ctx.scratch.get("n", 0) + 1
            return await call_next(tool, args)

        async def after_run(self, ctx, result):
            seen.append(ctx.scratch.get("n", 0))
            return result

    agent = _agent(
        _tool_then(name="approve"),
        Tally(),
        tools=[FunctionTool(name="approve", fn=_approve)],
        checkpointer=SQLiteCheckpointer(db_path=str(tmp_path / "cp.db")),
    )
    await agent.arun("go", execution_id="h-scratch")
    await agent.aresume("h-scratch", resume_value=Resume(approved=True))
    assert seen == [1]


# ── wrap_model ─────────────────────────────────────────────────────────────


class Onion(AgentMiddleware):
    def __init__(self, tag: str, log: list[str]) -> None:
        self.tag, self.log = tag, log

    async def wrap_model(self, ctx, messages, call_next):
        self.log.append(f"{self.tag}>")
        response = await call_next(messages)
        self.log.append(f"<{self.tag}")
        return response


@pytest.mark.parametrize("mode", ["arun", "astream"])
@pytest.mark.asyncio
async def test_wrap_model_is_an_onion_around_every_model_call(mode: str) -> None:
    log: list[str] = []
    agent = _agent(_tool_then(), Onion("a", log), Onion("b", log))
    if mode == "arun":
        await agent.arun("go")
    else:
        await _stream(agent, "go")
    assert log == ["a>", "b>", "<b", "<a"] * 2


class Answer(BaseModel):
    value: int


def test_the_structured_reask_goes_through_wrap_model() -> None:
    log: list[str] = []
    llm = _text("not json", '{"value": 7}')
    agent = _agent(llm, Onion("a", log), output_type=Answer, tools=[])
    result = agent.run("seven?")
    assert result.parsed == Answer(value=7)
    assert log == ["a>", "<a"] * 2


def test_the_guardrail_reask_goes_through_wrap_model() -> None:
    log: list[str] = []
    rule = Guardrail(
        name="no-bad",
        position=GuardrailPosition.output,
        fn=lambda text: GuardrailResult(passed="bad" not in text, message="say it nicely"),
        action="reask",
    )
    agent = _agent(_text("bad answer", "good answer"), Onion("a", log), guardrails=[rule], tools=[])
    assert agent.run("q").output == "good answer"
    assert log == ["a>", "<a"] * 2


class Cache(AgentMiddleware):
    async def wrap_model(self, ctx, messages, call_next):
        return LLMResponse(content="from cache", finish_reason="stop")


@pytest.mark.parametrize("mode", ["arun", "astream"])
@pytest.mark.asyncio
async def test_wrap_model_can_answer_without_calling_the_model(mode: str) -> None:
    llm = _text("from the model")
    agent = _agent(llm, Cache())
    if mode == "arun":
        assert (await agent.arun("q")).output == "from cache"
    else:
        events = await _stream(agent, "q")
        assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "from cache"
    assert llm._call_count == 0


class RetryOnce(AgentMiddleware):
    async def wrap_model(self, ctx, messages, call_next):
        try:
            return await call_next(messages)
        except RuntimeError:
            return await call_next(messages)


class Flaky(MockLLMClient):
    async def acomplete(self, messages, tools=None, **kwargs):
        if self._call_count == 0:
            self._call_count += 1
            raise RuntimeError("503")
        return await super().acomplete(messages, tools, **kwargs)

    async def astream(self, messages, tools=None, **kwargs):
        if self._call_count == 0:
            self._call_count += 1
            raise RuntimeError("503")
        async for event in super().astream(messages, tools, **kwargs):
            yield event


@pytest.mark.parametrize("mode", ["arun", "astream"])
@pytest.mark.asyncio
async def test_wrap_model_can_retry_a_failed_call(mode: str) -> None:
    # The first call fails (and uses up response 0); the retry gets response 1.
    llm = Flaky(responses=[LLMResponse(content="unused"), LLMResponse(content="second try")])
    agent = _agent(llm, RetryOnce(), tools=[])
    if mode == "arun":
        assert (await agent.arun("q")).output == "second try"
    else:
        events = await _stream(agent, "q")
        assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "second try"


@pytest.mark.parametrize("mode", ["arun", "astream"])
@pytest.mark.asyncio
async def test_ctx_llm_swaps_the_model_for_one_call(mode: str) -> None:
    main, cheap = _tool_then(final="main answer"), _text("cheap answer")

    class FinalTurnOnCheapModel(AgentMiddleware):
        async def before_model(self, ctx, messages):
            if ctx.turn == 1:
                ctx.llm = cheap
            return messages

    agent = _agent(main, FinalTurnOnCheapModel())
    if mode == "arun":
        output = (await agent.arun("go")).output
    else:
        events = await _stream(agent, "go")
        output = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert output == "cheap answer"
    assert (main._call_count, cheap._call_count) == (1, 1)
    assert agent.llm is main  # one call, not the agent


@pytest.mark.parametrize("mode", ["arun", "astream"])
@pytest.mark.asyncio
async def test_ctx_tools_changes_what_one_call_offers(mode: str) -> None:
    def lookup(text: str) -> str:
        """Look something up."""
        return f"found:{text}"

    extra = FunctionTool(name="lookup", fn=lookup)

    class OfferLookupInstead(AgentMiddleware):
        async def before_model(self, ctx, messages):
            if ctx.turn == 0:
                ctx.tools = [extra]
            return messages

    llm = _tool_then(name="lookup")
    agent = _agent(llm, OfferLookupInstead())
    if mode == "arun":
        result = await agent.arun("go")
        assert result.tool_calls[0]["tool_name"] == "lookup"
    else:
        await _stream(agent, "go")
    offered = [[t["function"]["name"] for t in (c["tools"] or [])] for c in llm._calls]
    assert offered == [["lookup"], ["echo"]]
    assert [m.content for m in llm._calls[1]["messages"] if m.role.value == "tool"] == ["found:hi"]


@pytest.mark.asyncio
async def test_ctx_carries_the_execution_id_and_the_runs_usage() -> None:
    seen: list[tuple[str, int, int]] = []

    class Peek(AgentMiddleware):
        async def before_model(self, ctx, messages):
            seen.append((ctx.execution_id, ctx.usage.tokens, ctx.usage.calls))
            return messages

    class Priced(MockLLMClient):
        async def acomplete(self, messages, tools=None, **kwargs):
            from fastaiagent._internal.pricing import record_run_tokens

            response = await super().acomplete(messages, tools, **kwargs)
            record_run_tokens(response.usage)
            return response

    llm = Priced(responses=_tool_then()._responses)
    await _agent(llm, Peek()).arun("go", execution_id="ex-usage")
    assert [s[0] for s in seen] == ["ex-usage", "ex-usage"]
    assert [s[1] for s in seen] == [0, 15]


# ── on_error ───────────────────────────────────────────────────────────────


class Down(MockLLMClient):
    async def acomplete(self, messages, tools=None, **kwargs):
        raise RuntimeError("provider down")

    async def astream(self, messages, tools=None, **kwargs):
        raise RuntimeError("provider down")
        yield  # pragma: no cover


@pytest.mark.parametrize("mode", ["arun", "astream"])
@pytest.mark.asyncio
async def test_on_error_sees_the_failure_once_and_cannot_swallow_it(mode: str) -> None:
    mw = Count()
    agent = _agent(Down(), mw)
    with pytest.raises(RuntimeError, match="provider down"):
        if mode == "arun":
            await agent.arun("go")
        else:
            await _stream(agent, "go")
    assert [str(e) for e in mw.errors] == ["provider down"]
    assert mw.after == 0


@pytest.mark.asyncio
async def test_a_raising_on_error_never_replaces_the_runs_error(caplog) -> None:
    class Broken(AgentMiddleware):
        async def on_error(self, ctx, error):
            raise ValueError("hook bug")

    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="provider down"):
        await _agent(Down(), Broken()).arun("go")
    assert "hook bug" in caplog.text


@pytest.mark.asyncio
async def test_a_stop_and_a_pause_are_not_errors(tmp_path) -> None:
    class StopInTool(AgentMiddleware):
        async def wrap_tool(self, ctx, tool, args, call_next):
            raise StopAgent("budget", reason="budget")

    stopped = Count()
    await _agent(_tool_then(), stopped, StopInTool()).arun("go")
    assert stopped.errors == [] and stopped.after == 1

    paused = Count()
    await _agent(
        _tool_then(name="approve"),
        paused,
        tools=[FunctionTool(name="approve", fn=_approve)],
        checkpointer=SQLiteCheckpointer(db_path=str(tmp_path / "cp.db")),
    ).arun("go")
    assert paused.errors == []


@pytest.mark.asyncio
async def test_on_error_fires_when_a_resume_fails(tmp_path) -> None:
    """The phase-0 X2 path: the re-invoked call is blocked on the way back in."""
    n = {"calls": 0}

    def second_time_blocks(text: str) -> GuardrailResult:
        n["calls"] += 1
        return GuardrailResult(passed=n["calls"] < 2, message="blocked on resume")

    mw = Count()
    agent = _agent(
        _tool_then(name="approve"),
        mw,
        tools=[FunctionTool(name="approve", fn=_approve)],
        guardrails=[
            Guardrail(name="g", position=GuardrailPosition.tool_call, fn=second_time_blocks)
        ],
        checkpointer=SQLiteCheckpointer(db_path=str(tmp_path / "cp.db")),
    )
    await agent.arun("go", execution_id="h-err")
    with pytest.raises(Exception, match="blocked on resume"):
        await agent.aresume("h-err", resume_value=Resume(approved=True))
    assert [str(e) for e in mw.errors] == ["blocked on resume"]


# ── on_stream_event ────────────────────────────────────────────────────────


class Shout(AgentMiddleware):
    async def on_stream_event(self, ctx, event):
        if isinstance(event, TextDelta):
            return TextDelta(text=event.text.upper())
        if isinstance(event, Usage):
            return None
        return event


@pytest.mark.asyncio
async def test_on_stream_event_rewrites_and_drops_events() -> None:
    events = await _stream(_agent(_tool_then(final="done"), Shout()), "go")
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "DONE"
    assert not any(isinstance(e, Usage) for e in events)


@pytest.mark.asyncio
async def test_on_stream_event_cannot_drop_the_end_or_a_pause(tmp_path) -> None:
    class DropEverything(AgentMiddleware):
        async def on_stream_event(self, ctx, event):
            return None

    done = await _stream(_agent(_text("hi"), DropEverything()), "go")
    assert [type(e) for e in done] == [StreamDone]

    paused = await _stream(
        _agent(
            _tool_then(name="approve"),
            DropEverything(),
            tools=[FunctionTool(name="approve", fn=_approve)],
            checkpointer=SQLiteCheckpointer(db_path=str(tmp_path / "cp.db")),
        ),
        "go",
    )
    assert [type(e) for e in paused] == [Paused]


# ── 1E: Reflect's fail-open is recorded, not silent ────────────────────────


def test_reflect_records_that_it_skipped(caplog) -> None:
    from fastaiagent.agent.middleware import Reflect

    seen: dict[str, Any] = {}

    class Keep(AgentMiddleware):
        async def after_run(self, ctx, result):
            seen.update(ctx.scratch)
            return result

    reviewer = _text("this is not json")
    with caplog.at_level(logging.WARNING):
        result = _agent(_text("draft"), Keep(), Reflect(llm=reviewer), tools=[]).run("q")
    assert result.output == "draft"  # fail-open: the draft stands
    assert seen.get("reflect_skipped") == 1
    assert "Reflect" in caplog.text and "kept the draft" in caplog.text
