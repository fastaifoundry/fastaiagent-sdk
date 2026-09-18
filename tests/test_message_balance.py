"""The tool-call/tool-result invariant, owned as a sweep.

The provider contract both OpenAI and Anthropic enforce on every request:

  * every assistant message carrying ``tool_calls`` is followed by **exactly
    one** tool result per ``tool_call_id``;
  * every tool result has a parent tool call.

Violate it and the next request 400s — *if* one happens. When it doesn't, the
history is quietly wrong and the next thing to re-send it (a structured-output
re-ask, a ``reask`` guardrail, a resume, a fork) is what pays. Before 1.67.0
five code paths could leave the list unbalanced and **nothing in ``tests/``
asserted the invariant at all**: ``test_middleware.py`` covered the ``StopAgent``
hook that is *not* broken, and ``test_parallel_tools.py`` asserted ordering but
never balance.

This file owns the invariant as a property rather than a per-path regression
test, in the spirit of ``test_guardrail_unusable_config_sweep.py``: a new early
return that leaves the list unbalanced fails here even though it is not named
here. Assertions are on the message list the agent actually hands the provider —
captured live from the LLM client — never on ``result.output``, which stays
identical whether the history is balanced or not.

No mocks of the thing under test: real ``Agent``, real middleware, real
``SQLiteCheckpointer``. ``MockLLMClient`` is the repo's own ``LLMClient``
subclass (see ``tests/conftest.py``) — what the model *says* is irrelevant here,
what the agent *sends* is the whole subject.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import BaseModel

from fastaiagent import Agent, SQLiteCheckpointer, ToolBudget, TrimLongMessages
from fastaiagent.agent.middleware import MiddlewareContext
from fastaiagent.chain.checkpoint import is_run_end
from fastaiagent.chain.interrupt import Resume, interrupt
from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition, GuardrailType
from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.llm.message import (
    AssistantMessage,
    Message,
    MessageRole,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from fastaiagent.tool.function import FunctionTool
from tests.conftest import MockLLMClient

SSN = r"\b\d{3}-\d{2}-\d{4}\b"

# --------------------------------------------------------------------------- #
# The invariant, spelled out once
# --------------------------------------------------------------------------- #


def imbalance(messages: list[Message]) -> tuple[list[str], list[str]]:
    """Return ``(unanswered_tool_call_ids, orphaned_tool_result_ids)``.

    Deliberately re-implemented here rather than imported from the code under
    test: a test that calls the production checker certifies rather than checks.
    """
    unanswered: list[str] = []
    orphans: list[str] = []
    open_ids: list[str] = []
    answered: set[str] = set()

    def close() -> None:
        nonlocal open_ids, answered
        unanswered.extend([i for i in open_ids if i not in answered])
        open_ids, answered = [], set()

    for m in messages:
        if m.role == MessageRole.assistant and m.tool_calls:
            close()
            open_ids = [tc.id for tc in m.tool_calls]
        elif m.role == MessageRole.tool:
            tcid = m.tool_call_id or ""
            if tcid in open_ids and tcid not in answered:
                answered.add(tcid)
            else:
                orphans.append(tcid)
        else:
            close()
    close()
    return unanswered, orphans


def assert_balanced(messages: list[Message], where: str) -> None:
    unanswered, orphans = imbalance(messages)
    shape = [
        (m.role.value, [tc.id for tc in (m.tool_calls or [])] or m.tool_call_id or "")
        for m in messages
    ]
    assert not unanswered, (
        f"{where}: tool_call(s) {unanswered} were never answered. History: {shape}"
    )
    assert not orphans, (
        f"{where}: tool result(s) {orphans} have no parent tool_call. History: {shape}"
    )


def test_the_checker_itself_catches_both_shapes() -> None:
    """A checker that cannot fail would make every test below vacuous."""
    a = AssistantMessage(tool_calls=[ToolCall(id="t1", name="x", arguments={})])
    assert imbalance([UserMessage("u"), a]) == (["t1"], [])
    assert imbalance([UserMessage("u"), ToolMessage("r", "t9")]) == ([], ["t9"])
    assert imbalance([UserMessage("u"), a, ToolMessage("r", "t1")]) == ([], [])
    # Two results for one call is also a violation: "exactly one".
    assert imbalance([a, ToolMessage("r", "t1"), ToolMessage("r", "t1")]) == ([], ["t1"])


# --------------------------------------------------------------------------- #
# Fixtures: a recording client and a couple of tools
# --------------------------------------------------------------------------- #


class RecordingLLM(MockLLMClient):
    """``MockLLMClient`` that snapshots what it was handed.

    ``sent`` holds a deep copy per call — the history as the provider would have
    received it at that moment. ``live`` holds the list *object*, which is the
    one the executor keeps appending to, so a path that ends without ever making
    another LLM call is still observable.
    """

    def __init__(self, responses: list[LLMResponse]):
        super().__init__(responses=responses)
        self.sent: list[list[Message]] = []
        self.live: list[list[Message]] = []

    async def acomplete(
        self, messages: list[Message], tools: Any = None, **kwargs: Any
    ) -> LLMResponse:
        self.sent.append(copy.deepcopy(messages))
        self.live.append(messages)
        return await super().acomplete(messages, tools=tools, **kwargs)


def _echo_tool(calls: list[str] | None = None) -> FunctionTool:
    async def echo(text: str) -> str:
        if calls is not None:
            calls.append(text)
        return f"echoed:{text}"

    return FunctionTool(
        name="echo",
        fn=echo,
        description="Echo text back",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )


def _two_calls_then(final: str) -> list[LLMResponse]:
    """One assistant turn carrying two tool calls, then a final text."""
    return [
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(id="t1", name="echo", arguments={"text": "one"}),
                ToolCall(id="t2", name="echo", arguments={"text": "two"}),
            ],
            finish_reason="tool_calls",
        ),
        LLMResponse(content=final, finish_reason="stop"),
    ]


# --------------------------------------------------------------------------- #
# (a) StopAgent raised from wrap_tool, mid-turn
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stop_agent_from_wrap_tool_leaves_no_unanswered_tool_calls() -> None:
    """``ToolBudget`` stops on the second of two calls in the same turn.

    The assistant message already declares both. Returning here without
    answering ``t2`` leaves a history no provider will accept.
    """
    llm = RecordingLLM(_two_calls_then("unreached"))
    agent = Agent(
        name="stopper",
        llm=llm,
        tools=[_echo_tool()],
        middleware=[ToolBudget(max_calls=1, message="over budget")],
    )
    await agent.arun("go", trace=False)

    assert_balanced(llm.live[-1], "history after a wrap_tool StopAgent")


@pytest.mark.asyncio
async def test_a_skipped_sibling_is_answered_not_re_dispatched() -> None:
    """The synthetic result must say so — and the tool must not have run."""
    calls: list[str] = []
    llm = RecordingLLM(_two_calls_then("unreached"))
    agent = Agent(
        name="stopper",
        llm=llm,
        tools=[_echo_tool(calls)],
        middleware=[ToolBudget(max_calls=1, message="over budget")],
    )
    await agent.arun("go", trace=False)

    assert calls == ["one"], "the skipped sibling must not be dispatched"
    history = llm.live[-1]
    answers = {m.tool_call_id: m.content for m in history if m.role == MessageRole.tool}
    assert set(answers) == {"t1", "t2"}
    assert "echoed:one" in str(answers["t1"])
    assert "not run" in str(answers["t2"]).lower()


# --------------------------------------------------------------------------- #
# (e) the consumers that turn (a) into a 400
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_reask_guardrail_re_sends_a_balanced_history_without_output_type() -> None:
    """No ``output_type`` anywhere — this is not a structured-output bug."""
    llm = RecordingLLM(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="t1", name="echo", arguments={"text": "one"}),
                    ToolCall(id="t2", name="echo", arguments={"text": "two"}),
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="clean", finish_reason="stop"),
        ]
    )
    rule = Guardrail(
        name="no-ssn",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.output,
        config={"pattern": SSN},
        action="reask",
    )
    agent = Agent(
        name="reasker",
        llm=llm,
        tools=[_echo_tool()],
        guardrails=[rule],
        # The stop message is what becomes the output, so make it trip the rule.
        middleware=[ToolBudget(max_calls=1, message="stopping on 123-45-6789")],
    )
    result = await agent.arun("go", trace=False)

    assert result.output == "clean", "the re-ask turn must actually have happened"
    assert len(llm.sent) == 2, "expected exactly one re-ask call"
    assert_balanced(llm.sent[-1], "history re-sent by a reask guardrail")


class _Answer(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_structured_output_reask_re_sends_a_balanced_history() -> None:
    llm = RecordingLLM(_two_calls_then('{"answer": "ok"}'))
    agent = Agent(
        name="structured",
        llm=llm,
        tools=[_echo_tool()],
        output_type=_Answer,
        middleware=[ToolBudget(max_calls=1, message="over budget")],
    )
    await agent.arun("go", trace=False)

    assert len(llm.sent) == 2, "expected a structured-output re-ask"
    assert_balanced(llm.sent[-1], "history re-sent by _reask_structured")


# --------------------------------------------------------------------------- #
# (d) TrimLongMessages
# --------------------------------------------------------------------------- #


def _history_ending_in_tool_results() -> list[Message]:
    return [
        SystemMessage("s"),
        UserMessage("u"),
        AssistantMessage(
            tool_calls=[
                ToolCall(id="t1", name="echo", arguments={}),
                ToolCall(id="t2", name="echo", arguments={}),
            ]
        ),
        ToolMessage("r1", "t1"),
        ToolMessage("r2", "t2"),
    ]


def _history_ending_in_text() -> list[Message]:
    return [*_history_ending_in_tool_results(), AssistantMessage(content="done")]


@pytest.mark.parametrize("keep_last", [1, 2, 3, 4, 5])
@pytest.mark.parametrize(
    "build", [_history_ending_in_tool_results, _history_ending_in_text], ids=["tail", "text"]
)
@pytest.mark.asyncio
async def test_trimming_never_orphans_a_tool_result(build, keep_last: int) -> None:
    """A shipped built-in must not cut between a tool call and its result."""
    out = await TrimLongMessages(keep_last=keep_last).before_model(MiddlewareContext(), build())
    assert_balanced(out, f"TrimLongMessages(keep_last={keep_last})")


@pytest.mark.asyncio
async def test_trimming_still_trims_and_still_keeps_the_system_message() -> None:
    """Balancing must not quietly turn the trimmer into a no-op."""
    mw = TrimLongMessages(keep_last=5)
    messages: list[Message] = [SystemMessage("system")]
    for i in range(30):
        messages.append(UserMessage(f"msg-{i}"))
    out = await mw.before_model(MiddlewareContext(), messages)
    assert len(out) == 6
    assert out[0].role == MessageRole.system
    assert out[-1].content == "msg-29"


# --------------------------------------------------------------------------- #
# (c) resume after an interrupt on the first of several parallel calls
# --------------------------------------------------------------------------- #


def _interrupting_tool() -> FunctionTool:
    async def needs_approval(amount: int) -> dict:
        decision = interrupt(reason="approve?", context={"amount": amount})
        return {"approved": bool(getattr(decision, "approved", decision))}

    return FunctionTool(
        name="needs_approval",
        fn=needs_approval,
        description="Ask a human",
        parameters={
            "type": "object",
            "properties": {"amount": {"type": "integer"}},
            "required": ["amount"],
        },
    )


def _sibling_tool(calls: list[int]) -> FunctionTool:
    async def sibling(amount: int) -> dict:
        calls.append(amount)
        return {"sent": amount}

    return FunctionTool(
        name="sibling",
        fn=sibling,
        description="A side-effecting sibling",
        parameters={
            "type": "object",
            "properties": {"amount": {"type": "integer"}},
            "required": ["amount"],
        },
    )


@pytest.mark.asyncio
async def test_resume_after_interrupt_on_the_first_of_parallel_calls(tmp_path) -> None:
    """The persisted shape — this one survives the process that made it."""
    sibling_calls: list[int] = []
    llm = RecordingLLM(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="t1", name="needs_approval", arguments={"amount": 1}),
                    ToolCall(id="t2", name="sibling", arguments={"amount": 2}),
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="done", finish_reason="stop"),
        ]
    )
    store = SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))
    store.setup()
    agent = Agent(
        name="resumer",
        llm=llm,
        tools=[_interrupting_tool(), _sibling_tool(sibling_calls)],
        checkpointer=store,
    )

    paused = await agent.arun("go", execution_id="ex-parallel", trace=False)
    assert paused.status == "paused"
    assert sibling_calls == [], "the sibling never ran before the pause"

    done = await agent.aresume("ex-parallel", resume_value=Resume(approved=True))
    assert done.status == "completed"

    assert_balanced(llm.sent[-1], "history re-sent after a resume")
    assert sibling_calls == [], (
        "a sibling skipped by a resume must NOT be re-dispatched — the resume "
        "machinery cannot re-enter mid-turn, and firing it here would be a "
        "second side effect the model never saw a first result for"
    )
    answers = {m.tool_call_id: m.content for m in llm.sent[-1] if m.role == MessageRole.tool}
    assert set(answers) == {"t1", "t2"}
    assert "not run" in str(answers["t2"]).lower()


# --------------------------------------------------------------------------- #
# The adjacent durability defect: a failure AFTER the tool loop
# --------------------------------------------------------------------------- #


def _charging_tool(charges: list[int]) -> FunctionTool:
    async def charge(amount: int) -> dict:
        charges.append(amount)
        return {"charged": amount}

    return FunctionTool(
        name="charge",
        fn=charge,
        description="Charge a card",
        parameters={
            "type": "object",
            "properties": {"amount": {"type": "integer"}},
            "required": ["amount"],
        },
    )


def _blocking_rule() -> Guardrail:
    return Guardrail(
        name="no-ssn",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.output,
        config={"pattern": SSN},
        action="block",
    )


def _post_loop_failure_agent(store, charges: list[int]) -> Agent:
    """An agent that dies in its OUTPUT guardrail — after the tool loop.

    ``ToolBudget`` ends the loop cooperatively on the second turn's call, so the
    newest checkpoint is that turn's *pre-tool* row. The stop message then trips
    a blocking output guardrail, which raises from outside the loop.
    """
    return Agent(
        name="charger",
        llm=MockLLMClient(
            responses=[
                LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="t1", name="charge", arguments={"amount": 1})],
                    finish_reason="tool_calls",
                ),
                LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="t2", name="charge", arguments={"amount": 2})],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="clean", finish_reason="stop"),
            ]
        ),
        tools=[_charging_tool(charges)],
        checkpointer=store,
        guardrails=[_blocking_rule()],
        middleware=[ToolBudget(max_calls=1, message="stopping on 123-45-6789")],
    )


@pytest.mark.asyncio
async def test_a_post_loop_failure_writes_a_failed_run_end_marker(tmp_path) -> None:
    """1.65.0's D5 marker only wrapped ``execute_tool_loop``.

    Everything after it — the re-ask, the output guardrails, memory — sat
    outside the ``try``, so a run that died there left no tombstone at all: the
    newest row was a ``completed`` step checkpoint, identical to a run that had
    simply finished.
    """
    charges: list[int] = []
    store = SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))
    store.setup()

    with pytest.raises(Exception):
        await _post_loop_failure_agent(store, charges).arun(
            "go", execution_id="ex-post", trace=False
        )

    markers = [c for c in store.list("ex-post") if is_run_end(c)]
    assert len(markers) == 1, (
        f"expected exactly one run-end row for a run that died after the tool "
        f"loop, got {len(markers)}"
    )
    assert markers[0].status == "failed"
    assert "Pattern matched" in markers[0].state_snapshot.get("run_error", "")


@pytest.mark.asyncio
async def test_a_resume_after_a_post_loop_failure_does_not_re_fire_the_tool(
    tmp_path,
) -> None:
    """The run died *after* the loop, so the loop is not the re-entry point.

    Without the loop-end checkpoint the newest resumable row is the pre-tool
    one, and ``aresume`` re-invokes the tool — one charge becomes two.
    """
    charges: list[int] = []
    store = SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))
    store.setup()
    agent = _post_loop_failure_agent(store, charges)

    with pytest.raises(Exception):
        await agent.arun("go", execution_id="ex-refire", trace=False)
    assert charges == [1]

    await agent.aresume("ex-refire")
    assert charges == [1], f"the tool re-fired on resume: {charges}"


# --------------------------------------------------------------------------- #
# (b) the streaming twin
# --------------------------------------------------------------------------- #


class RecordingStreamLLM(MockLLMClient):
    """Same idea as ``RecordingLLM`` for the ``astream`` path."""

    def __init__(self, responses: list[LLMResponse]):
        super().__init__(responses=responses)
        self.live: list[list[Message]] = []

    async def astream(self, messages: list[Message], tools: Any = None, **kwargs: Any):
        self.live.append(messages)
        async for event in super().astream(messages, tools=tools, **kwargs):
            yield event


@pytest.mark.asyncio
async def test_streaming_stop_agent_leaves_no_unanswered_tool_calls() -> None:
    """``stream_tool_loop`` had the identical bare ``return``."""
    llm = RecordingStreamLLM(_two_calls_then("unreached"))
    agent = Agent(
        name="streamer",
        llm=llm,
        tools=[_echo_tool()],
        middleware=[ToolBudget(max_calls=1, message="over budget")],
    )
    async for _ in agent.astream("go", trace=False):
        pass

    assert_balanced(llm.live[-1], "history after a streaming wrap_tool StopAgent")


# --------------------------------------------------------------------------- #
# The provider boundary: repair loudly, never silently
# --------------------------------------------------------------------------- #


class _CapturingClient(LLMClient):
    """A real ``LLMClient`` with only the HTTP call swapped out.

    Overriding ``_get_provider_fn`` (and not ``acomplete``) keeps the whole
    boundary — including the sanitize — on the path under test.
    """

    def __init__(self) -> None:
        super().__init__(provider="openai", model="gpt-4o-mini", api_key="test")
        self.seen: list[list[Message]] = []

    def _get_provider_fn(self) -> Any:
        async def _fn(messages: list[Message], tools: Any = None, **kwargs: Any) -> LLMResponse:
            self.seen.append(list(messages))
            return LLMResponse(content="ok", finish_reason="stop")

        return _fn


@pytest.mark.asyncio
async def test_the_provider_boundary_repairs_a_broken_history_and_says_so(caplog) -> None:
    """Belt and braces for paths nobody has found, and hand-built histories.

    The repair must be **loud**: a silent one turns every future instance of
    this class of bug into an invisible fix.
    """
    client = _CapturingClient()
    broken = [
        UserMessage("u"),
        AssistantMessage(
            tool_calls=[
                ToolCall(id="t1", name="echo", arguments={}),
                ToolCall(id="t2", name="echo", arguments={}),
            ]
        ),
        ToolMessage("r1", "t1"),
        ToolMessage("stray", "t99"),
    ]
    with caplog.at_level("WARNING", logger="fastaiagent.llm.client"):
        await client.acomplete(broken)

    assert_balanced(client.seen[-1], "history at the provider boundary")
    warning = "\n".join(r.getMessage() for r in caplog.records if r.levelname == "WARNING")
    assert "t2" in warning, f"the unanswered id must be named: {warning!r}"
    assert "t99" in warning, f"the orphaned id must be named: {warning!r}"

    # The caller's own list is untouched — the repair is per-request.
    assert len(broken) == 4


@pytest.mark.asyncio
async def test_the_provider_boundary_is_silent_on_a_balanced_history(caplog) -> None:
    client = _CapturingClient()
    with caplog.at_level("WARNING", logger="fastaiagent.llm.client"):
        await client.acomplete(
            [
                UserMessage("u"),
                AssistantMessage(tool_calls=[ToolCall(id="t1", name="echo", arguments={})]),
                ToolMessage("r1", "t1"),
            ]
        )
    # Filter to this warning specifically — unrelated client warnings can leak
    # in from whatever ran before us in the session.
    assert [r for r in caplog.records if "tool-call contract" in r.getMessage()] == []
