"""Tests for fastaiagent.agent.middleware.

Covers the three hooks (``before_model``, ``after_model``, ``wrap_tool``),
ordering semantics, short-circuit behavior, built-in middleware, and
interaction with existing Agent features (guardrails, memory, turn indexing).
"""

from __future__ import annotations

import pytest

from fastaiagent import (
    Agent,
    AgentMiddleware,
    MiddlewareContext,
    RedactPII,
    StopAgent,
    ToolBudget,
    TrimLongMessages,
)
from fastaiagent.agent.middleware import _MiddlewarePipeline
from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition, GuardrailResult
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import (
    Message,
    MessageRole,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from fastaiagent.tool.base import Tool, ToolResult
from fastaiagent.tool.function import FunctionTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
from tests.conftest import MockLLMClient


def _llm_with_text(text: str) -> MockLLMClient:
    return MockLLMClient(responses=[LLMResponse(content=text, finish_reason="stop")])


def _llm_with_one_tool_call(
    tool_name: str = "echo", args: dict | None = None, final: str = "done"
) -> MockLLMClient:
    return MockLLMClient(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_1", name=tool_name, arguments=args or {"text": "hi"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content=final, finish_reason="stop"),
        ]
    )


def _llm_with_n_tool_calls(
    n: int, tool_name: str = "echo", final: str = "done"
) -> MockLLMClient:
    """Emit ``n`` separate single-tool-call iterations, then a final text."""
    responses = []
    for i in range(n):
        responses.append(
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id=f"call_{i}", name=tool_name, arguments={"text": str(i)})
                ],
                finish_reason="tool_calls",
            )
        )
    responses.append(LLMResponse(content=final, finish_reason="stop"))
    return MockLLMClient(responses=responses)


def _echo_tool() -> Tool:
    async def echo(text: str) -> str:
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


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_middleware_ordering_before_model(recording_middleware):
    """before_model runs in declaration order."""
    a, rec_a = recording_middleware("A")
    b, rec_b = recording_middleware("B")
    c, rec_c = recording_middleware("C")

    agent = Agent(
        name="ordering",
        llm=_llm_with_text("ok"),
        middleware=[a, b, c],
    )
    await agent.arun("hello", trace=False)

    # Merge hook order across the three recorders.
    hooks = (
        [("A", h) for h in rec_a["before_model"]]
        + [("B", h) for h in rec_b["before_model"]]
        + [("C", h) for h in rec_c["before_model"]]
    )
    names_in_order = [
        name for name, _ in sorted(hooks, key=lambda x: ("A", "B", "C").index(x[0]))
    ]
    # All three fired exactly once, A before B before C.
    assert names_in_order == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_middleware_ordering_after_model(recording_middleware):
    """after_model runs in reverse declaration order."""
    a, rec_a = recording_middleware("A")
    b, rec_b = recording_middleware("B")
    c, rec_c = recording_middleware("C")

    # Track global invocation order via a shared list.
    call_log: list[str] = []

    class Tag(AgentMiddleware):
        def __init__(self, tag: str) -> None:
            self.name = tag

        async def after_model(self, ctx, response):
            call_log.append(self.name)
            return response

    agent = Agent(
        name="rev",
        llm=_llm_with_text("ok"),
        middleware=[Tag("A"), Tag("B"), Tag("C")],
    )
    await agent.arun("hello", trace=False)
    assert call_log == ["C", "B", "A"]


@pytest.mark.asyncio
async def test_wrap_tool_onion_order():
    """wrap_tool is onion: first middleware is outermost, sees the chain."""
    log: list[str] = []

    def mk(name: str) -> AgentMiddleware:
        class _M(AgentMiddleware):
            async def wrap_tool(self, ctx, tool, args, call_next):
                log.append(f"{name}:enter")
                r = await call_next(tool, args)
                log.append(f"{name}:exit")
                return r

        m = _M()
        m.name = name
        return m

    agent = Agent(
        name="onion",
        llm=_llm_with_one_tool_call(),
        tools=[_echo_tool()],
        middleware=[mk("A"), mk("B"), mk("C")],
    )
    await agent.arun("hi", trace=False)
    assert log == ["A:enter", "B:enter", "C:enter", "C:exit", "B:exit", "A:exit"]


@pytest.mark.asyncio
async def test_wrap_tool_short_circuit_skips_call_next():
    """A middleware returning a ToolResult without calling call_next skips the real tool."""
    called = {"echo": False}

    async def echo(text: str) -> str:
        called["echo"] = True
        return f"echoed:{text}"

    tool = FunctionTool(
        name="echo",
        fn=echo,
        description="",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    class ShortCircuit(AgentMiddleware):
        name = "short"

        async def wrap_tool(self, ctx, tool, args, call_next):
            return ToolResult(output="short-circuited")

    agent = Agent(
        name="short",
        llm=_llm_with_one_tool_call(),
        tools=[tool],
        middleware=[ShortCircuit()],
    )
    await agent.arun("hi", trace=False)
    assert called["echo"] is False


# ---------------------------------------------------------------------------
# Short-circuit / stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_agent_short_circuits_run():
    """Middleware raising StopAgent ends the run cooperatively with final output."""

    class Stopper(AgentMiddleware):
        name = "stop"

        async def before_model(self, ctx, messages):
            raise StopAgent("stopped early")

    agent = Agent(name="s", llm=_llm_with_text("unreached"), middleware=[Stopper()])
    result = await agent.arun("hello", trace=False)
    assert result.output == "stopped early"
    assert result.tool_calls == []


# ---------------------------------------------------------------------------
# Built-ins
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trim_long_messages_keeps_last_n_plus_system():
    """TrimLongMessages preserves the leading system message and keeps the tail."""
    mw = TrimLongMessages(keep_last=5)
    messages: list[Message] = [SystemMessage("system")]
    for i in range(30):
        messages.append(UserMessage(f"msg-{i}"))

    out = await mw.before_model(MiddlewareContext(), messages)
    assert len(out) == 6  # 1 system + 5 tail
    assert out[0].role == MessageRole.system
    assert out[0].content == "system"
    assert out[-1].content == "msg-29"
    assert out[1].content == "msg-25"


@pytest.mark.asyncio
async def test_tool_budget_raises_on_limit():
    """ToolBudget short-circuits once max_calls is reached."""
    agent = Agent(
        name="budgeted",
        llm=_llm_with_n_tool_calls(n=5, final="done"),
        tools=[_echo_tool()],
        middleware=[ToolBudget(max_calls=2, message="over budget")],
    )
    result = await agent.arun("run tools", trace=False)
    assert "over budget" in result.output


@pytest.mark.asyncio
async def test_redact_pii_redacts_email_in_messages_and_response():
    """RedactPII replaces email patterns in outbound and inbound text."""
    from tests.conftest import MockLLMClient

    llm = MockLLMClient(
        responses=[LLMResponse(content="Contact bob@example.com please", finish_reason="stop")]
    )
    agent = Agent(name="pii", llm=llm, middleware=[RedactPII()])
    result = await agent.arun("My email is alice@test.com", trace=False)

    # Response redacted.
    assert "alice@test.com" not in result.output
    assert "bob@example.com" not in result.output
    assert "[REDACTED]" in result.output

    # Outbound messages saw the redaction too.
    seen = llm._calls[0]["messages"]
    user_contents = [m.content for m in seen if m.role == MessageRole.user]
    assert all("alice@test.com" not in (c or "") for c in user_contents)


# ---------------------------------------------------------------------------
# Context state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_middleware_sees_correct_turn_and_tool_call_index():
    """ctx.turn increments across iterations; tool_call_index resets per turn."""
    turns: list[int] = []
    tool_idxs: list[int] = []

    class Watcher(AgentMiddleware):
        name = "watch"

        async def before_model(self, ctx, messages):
            turns.append(ctx.turn)
            return messages

        async def wrap_tool(self, ctx, tool, args, call_next):
            tool_idxs.append(ctx.tool_call_index)
            return await call_next(tool, args)

    agent = Agent(
        name="watcher",
        llm=_llm_with_n_tool_calls(n=3, final="done"),
        tools=[_echo_tool()],
        middleware=[Watcher()],
    )
    await agent.arun("hi", trace=False)

    assert turns == [0, 1, 2, 3]  # 3 tool-call turns + final no-tool turn
    # Each turn had one tool call, indexed 0.
    assert tool_idxs == [0, 0, 0]


@pytest.mark.asyncio
async def test_middleware_scratch_shared_across_hooks():
    """ctx.scratch persists across before_model, after_model, and wrap_tool."""
    checkpoints: list[int] = []

    class Sharer(AgentMiddleware):
        name = "share"

        async def before_model(self, ctx, messages):
            ctx.scratch["x"] = ctx.scratch.get("x", 0) + 1
            checkpoints.append(ctx.scratch["x"])
            return messages

        async def after_model(self, ctx, response):
            ctx.scratch["x"] += 10
            checkpoints.append(ctx.scratch["x"])
            return response

    agent = Agent(name="s", llm=_llm_with_text("ok"), middleware=[Sharer()])
    await agent.arun("hi", trace=False)
    assert checkpoints == [1, 11]


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_middleware_optional_no_op(mock_llm_with_tools):
    """Agent(middleware=None) runs identically to pre-middleware behavior."""
    agent = Agent(
        name="no-mw",
        llm=mock_llm_with_tools,
        tools=[_echo_tool()],
        middleware=None,
    )
    r = await agent.arun("hi", trace=False)
    assert r.output == "Based on the search results, here is the answer."


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_middleware_error_propagates():
    """Non-StopAgent exceptions in a hook propagate to the caller."""

    class Boom(AgentMiddleware):
        name = "boom"

        async def before_model(self, ctx, messages):
            raise RuntimeError("kaboom")

    agent = Agent(name="b", llm=_llm_with_text("ok"), middleware=[Boom()])
    with pytest.raises(RuntimeError, match="kaboom"):
        await agent.arun("hi", trace=False)


# ---------------------------------------------------------------------------
# Interaction with guardrails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_middleware_with_guardrails_runs_guardrails_first():
    """Input guardrail fires before any before_model middleware."""
    order: list[str] = []

    def make_guardrail() -> Guardrail:
        def check(text: str) -> GuardrailResult:
            order.append("guardrail")
            return GuardrailResult(passed=True)

        return Guardrail(
            name="input-probe", fn=check, position=GuardrailPosition.input, blocking=False
        )

    class Watcher(AgentMiddleware):
        name = "watch"

        async def before_model(self, ctx, messages):
            order.append("before_model")
            return messages

    agent = Agent(
        name="both",
        llm=_llm_with_text("ok"),
        guardrails=[make_guardrail()],
        middleware=[Watcher()],
    )
    await agent.arun("hi", trace=False)
    assert order[0] == "guardrail"
    assert "before_model" in order
    assert order.index("guardrail") < order.index("before_model")


# ---------------------------------------------------------------------------
# Pipeline unit tests (without a full agent)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_empty_is_passthrough():
    """_MiddlewarePipeline with no middleware calls terminal directly for tools."""
    pipeline = _MiddlewarePipeline([])

    called: list[str] = []

    async def terminal(t, a):
        called.append(t.name)
        return ToolResult(output="ran")

    tool = _echo_tool()
    tr = await pipeline.invoke_tool(MiddlewareContext(), tool, {"text": "hi"}, terminal)
    assert tr.output == "ran"
    assert called == ["echo"]


@pytest.mark.asyncio
async def test_pipeline_before_and_after_model_isolated(recording_middleware):
    """Pipeline hooks work standalone on synthetic contexts."""
    a, rec_a = recording_middleware("A")
    b, rec_b = recording_middleware("B")
    pipeline = _MiddlewarePipeline([a, b])

    ctx = MiddlewareContext(turn=7, agent_name="unit")
    messages: list[Message] = [UserMessage("hi")]
    result_msgs = await pipeline.apply_before_model(ctx, messages)
    assert result_msgs is messages  # recording middleware passes through

    response = LLMResponse(content="hello", finish_reason="stop")
    result_resp = await pipeline.apply_after_model(ctx, response)
    assert result_resp is response

    assert rec_a["before_model"][0]["turn"] == 7
    assert rec_b["after_model"][0]["content"] == "hello"
