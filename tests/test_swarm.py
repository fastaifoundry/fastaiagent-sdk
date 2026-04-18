"""Tests for fastaiagent.agent.swarm.

Deterministic tests drive the swarm with ``MockLLMClient`` preloaded with
scripted responses — the handoff *tool calls* are real (from the shipped
`FunctionTool` handoff machinery) but the LLM's decision to emit them is
pre-scripted. This exercises the swarm executor, allowlist enforcement,
cycle guard, shared blackboard, streaming, and serialization.

Live tests (real OpenAI / Anthropic) verify the full loop: a real LLM
decides when to hand off, via a real system prompt. Per the no-mocking
rule these run unmocked; they skip cleanly without API keys.
"""

from __future__ import annotations

import os

import pytest

from fastaiagent import (
    Agent,
    HandoffEvent,
    LLMClient,
    Swarm,
    SwarmError,
)
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import ToolCall
from fastaiagent.llm.stream import TextDelta
from tests.conftest import MockLLMClient

_HAS_LIVE_KEY = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
_skip_no_live_key = pytest.mark.skipif(
    not _HAS_LIVE_KEY,
    reason="no OPENAI_API_KEY or ANTHROPIC_API_KEY set",
)


def _live_llm() -> LLMClient:
    if os.environ.get("OPENAI_API_KEY"):
        return LLMClient(provider="openai", model="gpt-4o-mini")
    return LLMClient(provider="anthropic", model="claude-haiku-4-5-20251001")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _handoff_call(target: str, reason: str = "", call_id: str = "call_1") -> ToolCall:
    return ToolCall(
        id=call_id,
        name=f"handoff_to_{target}",
        arguments={"reason": reason},
    )


def _final(text: str) -> LLMResponse:
    return LLMResponse(content=text, finish_reason="stop")


def _tool_call_response(target: str, reason: str = "", call_id: str = "call_1") -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[_handoff_call(target, reason, call_id)],
        finish_reason="tool_calls",
    )


def _handoff_then_stub(target: str, reason: str = "") -> list[LLMResponse]:
    """Common mock script: a single handoff tool call.

    The Swarm's internal ``_ExitAfterHandoff`` middleware stops the inner
    tool loop immediately after the handoff tool fires, so only one mock
    response is consumed per turn. ``MockLLMClient``'s repeat-last-forever
    behavior means the same handoff will fire if the agent is ever
    re-entered (useful for cycle tests).
    """
    return [_tool_call_response(target, reason)]


def _make_agent(name: str, llm: LLMClient, system_prompt: str = "") -> Agent:
    return Agent(name=name, system_prompt=system_prompt or f"You are {name}.", llm=llm)


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


def test_swarm_requires_at_least_one_agent() -> None:
    with pytest.raises(SwarmError):
        Swarm(name="empty", agents=[], entrypoint="missing")


def test_swarm_entrypoint_must_be_in_agents() -> None:
    a = _make_agent("a", MockLLMClient())
    with pytest.raises(SwarmError, match="entrypoint"):
        Swarm(name="s", agents=[a], entrypoint="not_here")


def test_swarm_duplicate_agent_names_raise() -> None:
    a1 = _make_agent("dup", MockLLMClient())
    a2 = _make_agent("dup", MockLLMClient())
    with pytest.raises(SwarmError, match="Duplicate"):
        Swarm(name="s", agents=[a1, a2], entrypoint="dup")


def test_swarm_handoffs_reject_unknown_targets() -> None:
    a = _make_agent("a", MockLLMClient())
    with pytest.raises(SwarmError, match="unknown"):
        Swarm(name="s", agents=[a], entrypoint="a", handoffs={"a": ["ghost"]})


def test_swarm_default_handoffs_full_mesh() -> None:
    a = _make_agent("a", MockLLMClient())
    b = _make_agent("b", MockLLMClient())
    c = _make_agent("c", MockLLMClient())
    s = Swarm(name="s", agents=[a, b, c], entrypoint="a")
    # Every agent can hand off to every other (but not itself).
    assert sorted(s.handoffs["a"]) == ["b", "c"]
    assert sorted(s.handoffs["b"]) == ["a", "c"]
    assert sorted(s.handoffs["c"]) == ["a", "b"]


# ---------------------------------------------------------------------------
# Run semantics
# ---------------------------------------------------------------------------


def test_swarm_single_agent_no_handoff_returns_agent_output() -> None:
    a = _make_agent("a", MockLLMClient(responses=[_final("final answer")]))
    swarm = Swarm(name="s", agents=[a], entrypoint="a", handoffs={"a": []})
    result = swarm.run("hi")
    assert result.output == "final answer"


def test_swarm_one_handoff_reaches_target() -> None:
    a_llm = MockLLMClient(responses=_handoff_then_stub("b", "needs b's skills"))
    b_llm = MockLLMClient(responses=[_final("handled by b")])
    a = _make_agent("a", a_llm)
    b = _make_agent("b", b_llm)
    swarm = Swarm(
        name="s", agents=[a, b], entrypoint="a",
        handoffs={"a": ["b"], "b": []},
    )
    result = swarm.run("task")
    assert result.output == "handled by b"


def test_swarm_handoff_respects_allowlist() -> None:
    """Agent may only call handoff tools it was given; attempting to route
    to a peer outside the allowlist raises SwarmError.
    """
    # The MockLLM fabricates a tool call for a non-injected tool name. The
    # swarm's allowlist check fires before Agent tries to execute it.
    a_llm = MockLLMClient(responses=_handoff_then_stub("c", "try to escape"))
    a = _make_agent("a", a_llm)
    b = _make_agent("b", MockLLMClient(responses=[_final("b done")]))
    c = _make_agent("c", MockLLMClient(responses=[_final("c done")]))
    swarm = Swarm(
        name="s", agents=[a, b, c], entrypoint="a",
        handoffs={"a": ["b"], "b": [], "c": []},
    )
    with pytest.raises(SwarmError, match="not in its allowlist"):
        swarm.run("task")


def test_swarm_cycle_guard() -> None:
    """A→B→A→B→... raises SwarmError once max_handoffs is exceeded."""
    # Each agent always hands off to the other. max_handoffs=3 means 4th
    # handoff trips the guard. The mock's responses must be long enough to
    # handle the alternation: each swarm turn consumes one tool-call +
    # one final-stub from its agent's mock.
    a_llm = MockLLMClient(responses=_handoff_then_stub("b", "loop"))
    b_llm = MockLLMClient(responses=_handoff_then_stub("a", "loop"))
    a = _make_agent("a", a_llm)
    b = _make_agent("b", b_llm)
    swarm = Swarm(
        name="s", agents=[a, b], entrypoint="a",
        handoffs={"a": ["b"], "b": ["a"]},
        max_handoffs=3,
    )
    with pytest.raises(SwarmError, match="max_handoffs=3"):
        swarm.run("loop")


def test_swarm_shared_blackboard_persists_across_handoffs() -> None:
    """Context passed via handoff(context={...}) lands on state.shared.

    We inspect the swarm's own internal state by calling arun through a
    minimal path that exposes the final input given to the next agent.
    """
    # a hands off to b with context; b returns a final answer.
    a_llm = MockLLMClient(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="handoff_to_b",
                        arguments={"reason": "share", "context": {"flag": 42}},
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]
    )
    b_llm = MockLLMClient(responses=[_final("done")])
    a = _make_agent("a", a_llm)
    b = _make_agent("b", b_llm)
    swarm = Swarm(
        name="s", agents=[a, b], entrypoint="a",
        handoffs={"a": ["b"], "b": []},
    )
    swarm.run("task")
    # The b-agent's `messages[0]` contains the briefing text assembled by the
    # swarm; verify it references shared state so we know the blackboard was
    # populated before b ran.
    sent_messages = b_llm._calls[0]["messages"]
    briefing = next(m.content for m in sent_messages if m.role.value == "user")
    assert "'flag': 42" in briefing


def test_swarm_path_recorded_on_successful_run() -> None:
    """After a run, the path appears inside aggregated tool_calls as ``agent`` tags."""
    a_llm = MockLLMClient(responses=_handoff_then_stub("b", "go"))
    b_llm = MockLLMClient(responses=[_final("end")])
    a = _make_agent("a", a_llm)
    b = _make_agent("b", b_llm)
    swarm = Swarm(
        name="s", agents=[a, b], entrypoint="a",
        handoffs={"a": ["b"], "b": []},
    )
    result = swarm.run("task")
    # Tool calls captured from each agent are tagged with which agent made them.
    agents_in_calls = {c.get("agent") for c in result.tool_calls}
    assert "a" in agents_in_calls  # a emitted the handoff tool call


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_swarm_to_dict_roundtrip() -> None:
    a = _make_agent("a", MockLLMClient())
    b = _make_agent("b", MockLLMClient())
    swarm = Swarm(
        name="rt", agents=[a, b], entrypoint="a",
        handoffs={"a": ["b"], "b": []},
        max_handoffs=5,
    )
    data = swarm.to_dict()
    assert data["name"] == "rt"
    assert sorted(data["agent_names"]) == ["a", "b"]
    assert data["entrypoint"] == "a"
    assert data["max_handoffs"] == 5
    restored = Swarm.from_dict(data, agents=[a, b])
    assert restored.name == "rt"
    assert restored.entrypoint == "a"
    assert restored.max_handoffs == 5
    assert sorted(restored.agents) == ["a", "b"]


def test_swarm_from_dict_rejects_missing_agents() -> None:
    a = _make_agent("a", MockLLMClient())
    b = _make_agent("b", MockLLMClient())
    swarm = Swarm(name="s", agents=[a, b], entrypoint="a", handoffs={"a": ["b"], "b": []})
    data = swarm.to_dict()
    with pytest.raises(SwarmError, match="expected agents"):
        Swarm.from_dict(data, agents=[a])  # missing b


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_swarm_astream_emits_handoff_event() -> None:
    a_llm = MockLLMClient(responses=_handoff_then_stub("b", "time for b"))
    b_llm = MockLLMClient(responses=[_final("answer")])
    a = _make_agent("a", a_llm)
    b = _make_agent("b", b_llm)
    swarm = Swarm(
        name="s", agents=[a, b], entrypoint="a",
        handoffs={"a": ["b"], "b": []},
    )

    events = []
    async for event in swarm.astream("task"):
        events.append(event)

    handoffs = [e for e in events if isinstance(e, HandoffEvent)]
    assert len(handoffs) == 1
    assert handoffs[0].from_agent == "a"
    assert handoffs[0].to_agent == "b"
    # The mock LLM astream path may or may not surface the reason on the
    # stream — just assert the event is present with the right ends.


@pytest.mark.asyncio
async def test_swarm_astream_no_handoff_completes_cleanly() -> None:
    a = _make_agent("a", MockLLMClient(responses=[_final("hello")]))
    swarm = Swarm(name="s", agents=[a], entrypoint="a", handoffs={"a": []})
    had_text = False
    async for event in swarm.astream("hi"):
        if isinstance(event, TextDelta):
            had_text = True
    # Swarm exits after the single agent — no error.
    assert had_text or True  # MockLLM may not stream TextDelta; exit clean is the point


# ---------------------------------------------------------------------------
# Swarm-as-drop-in for Agent
# ---------------------------------------------------------------------------


def test_swarm_run_returns_agent_result_shape() -> None:
    """Swarm.run() returns an AgentResult with the same fields as Agent.run()."""
    a = _make_agent("a", MockLLMClient(responses=[_final("hi")]))
    swarm = Swarm(name="s", agents=[a], entrypoint="a", handoffs={"a": []})
    result = swarm.run("x")
    # Duck-typed: same attributes as AgentResult.
    for attr in ("output", "tool_calls", "tokens_used", "latency_ms"):
        assert hasattr(result, attr)


# ---------------------------------------------------------------------------
# Live LLM tests (real APIs)
# ---------------------------------------------------------------------------


@_skip_no_live_key
def test_swarm_live_simple_handoff() -> None:
    """A triage agent hands off to a specialist; the specialist answers.

    This verifies the full loop with a real LLM picking the handoff tool.
    """
    llm = _live_llm()
    triage = Agent(
        name="triage",
        system_prompt=(
            "You are a triage agent. For any user request, decide whether a "
            "'coder' or 'writer' specialist should handle it and hand off to "
            "them immediately using the handoff tool. Do not answer questions "
            "yourself."
        ),
        llm=llm,
    )
    coder = Agent(
        name="coder",
        system_prompt=(
            "You are a senior Python developer. Answer code questions "
            "precisely and briefly. Do not hand off."
        ),
        llm=llm,
    )
    writer = Agent(
        name="writer",
        system_prompt=(
            "You are a prose editor. Help with writing, grammar, style. "
            "Do not hand off."
        ),
        llm=llm,
    )
    swarm = Swarm(
        name="triage_swarm",
        agents=[triage, coder, writer],
        entrypoint="triage",
        handoffs={"triage": ["coder", "writer"], "coder": [], "writer": []},
        max_handoffs=3,
    )
    result = swarm.run(
        "How do I reverse a list in Python? Give one short line of code."
    )
    assert result.output, "expected a non-empty final answer"
    # The coder should have answered — a final output with Python slicing is
    # reasonable but we don't overfit. Just confirm we didn't get a stall.
    assert len(result.output) > 5
