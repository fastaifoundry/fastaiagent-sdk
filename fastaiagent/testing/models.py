"""Deterministic LLM stand-ins for tests.

These classes implement the ``LLMClient`` public surface (``acomplete``,
``astream``, ``complete``, ``stream``) but never make HTTP calls. They are
provider-tagged ``"test"`` so traces, replay, and the local UI render them
the same way they render any other model run.

Why this exists:
    - Users writing tests for their own agents need a deterministic LLM
      that doesn't burn API credit or fail on offline CI.
    - Mocking ``LLMClient`` directly forces every test to know about the
      class internals; ``TestModel`` is a drop-in replacement.

Trace span attributes use the existing ``set_genai_attributes`` helper so
the data shape matches real provider spans (``gen_ai.system="test"``).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.llm.decisions import DecisionResult
from fastaiagent.llm.message import Message, ToolCall
from fastaiagent.llm.stream import (
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
)

# How tool-call dicts are accepted by TestModel:
#   {"id": "call_1", "name": "search", "arguments": {"q": "..."}}
# (id is optional — auto-generated as call_<n> if absent)


@dataclass
class _CannedTurn:
    """One canned LLM turn — text and/or tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: tuple[int, int] = (0, 0)


def _coerce_tool_calls(
    tool_calls: list[dict[str, Any]] | list[ToolCall] | None,
    *,
    base_index: int = 0,
) -> list[ToolCall]:
    if not tool_calls:
        return []
    out: list[ToolCall] = []
    for i, tc in enumerate(tool_calls):
        if isinstance(tc, ToolCall):
            out.append(tc)
            continue
        # mypy can't narrow heterogeneous list unions inside a loop; assert
        # the runtime invariant.
        assert isinstance(tc, dict)
        out.append(
            ToolCall(
                id=tc.get("id") or f"call_{base_index + i}",
                name=tc["name"],
                arguments=dict(tc.get("arguments") or {}),
            )
        )
    return out


def _coerce_responses(
    response: str | list[str] | None,
    tool_calls: list[dict[str, Any]] | None,
    usage: tuple[int, int],
) -> list[_CannedTurn]:
    """Build the round-robin list of canned turns from constructor args."""
    if response is None and not tool_calls:
        return [_CannedTurn(text="ok", usage=usage)]
    if isinstance(response, list):
        if not response:
            return [_CannedTurn(text="ok", usage=usage)]
        return [_CannedTurn(text=t, usage=usage) for t in response]
    text = response if response is not None else ""
    tcs = _coerce_tool_calls(tool_calls, base_index=0)
    return [_CannedTurn(text=text, tool_calls=tcs, usage=usage)]


def _record_test_completion(
    *,
    model: str,
    messages: list[Message],
    tools: list[dict[str, Any]] | None,
    response: LLMResponse,
) -> None:
    """Emit a real OTel span tagged provider="test", and bill the run's tokens.

    Mirrors what ``LLMClient.acomplete`` does for real providers so the
    Local UI shows a span for ``TestModel`` runs (useful for snapshot
    replay tests of agent behaviour).

    The token count goes into the run-scoped accumulator for the same reason it
    does on the real client: these models implement the ``LLMClient`` surface
    rather than extend its internals, so they never reach
    ``_acomplete_with_retries`` — and without this an offline multi-turn run
    reports whichever turn came back last. Tokens only, deliberately: a
    ``test-model`` has no rate, and pricing it would flip ``cost_known`` to
    false on every offline run.
    """
    from fastaiagent._internal.pricing import record_run_tokens
    from fastaiagent.trace.otel import get_tracer
    from fastaiagent.trace.span import set_genai_attributes

    record_run_tokens(response.usage)

    tracer = get_tracer("fastaiagent.testing.models")
    with tracer.start_as_current_span(f"llm.test.{model}") as span:
        # Lazy json — avoid a hard dependency on the client's _serialize_for_span.
        import json as _json

        try:
            request_messages = _json.dumps(
                [m.to_openai_format() for m in messages], default=str
            )
        except Exception:
            request_messages = None
        try:
            request_tools = _json.dumps(tools, default=str) if tools else None
        except Exception:
            request_tools = None
        try:
            response_tool_calls = (
                _json.dumps(
                    [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ],
                    default=str,
                )
                if response.tool_calls
                else None
            )
        except Exception:
            response_tool_calls = None
        set_genai_attributes(
            span,
            system="test",
            model=model,
            request_messages=request_messages,
            request_tools=request_tools,
            input_tokens=response.usage.get("prompt_tokens"),
            output_tokens=response.usage.get("completion_tokens"),
            response_content=response.content,
            response_tool_calls=response_tool_calls,
            finish_reason=response.finish_reason or None,
        )
        # Attribute the call to the agent's registry prompt, as the real client
        # does, so an offline run shows up in the prompt's lineage too.
        from fastaiagent.prompt.provenance import stamp_prompt_provenance

        stamp_prompt_provenance(span)


DecisionsArg = Any  # DecisionResult | dict | list of either | callable(input, questions)


def _coerce_decision(value: Any, questions: list[Any], model: str) -> DecisionResult:
    """A canned decision as ``DecisionResult`` (accepts the wire dict too)."""
    from fastaiagent.llm.decisions import parse_decision

    if isinstance(value, DecisionResult):
        result = value.model_copy(deep=True)
    elif isinstance(value, dict):
        result = parse_decision({"model": model, **value}, questions)
    else:
        raise TypeError(
            "a canned decision must be a DecisionResult or its dict form "
            f"({{'answers': [...]}}); got {type(value).__name__}"
        )
    if not result.model:
        result.model = model
    return result


async def _test_decide(
    owner: Any,
    input: Any,
    questions: Any,
    produce: Callable[[Any, list[Any]], Any],
) -> DecisionResult:
    """Shared ``adecide`` for the offline models: canned answers, a real span.

    The span has the same name and attributes ``LLMClient.adecide`` writes, so
    a ``TestModel`` run captured into a trace replays under
    ``determinism="recorded"`` exactly like a live one. Tokens only — no price,
    for the same reason ``_record_test_completion`` gives.
    """
    from fastaiagent._internal.pricing import record_run_tokens
    from fastaiagent.llm.client import _serialize_for_span
    from fastaiagent.llm.decisions import normalize_questions, summarize_decision_input
    from fastaiagent.trace.otel import get_tracer
    from fastaiagent.trace.span import set_fastaiagent_attributes, set_genai_attributes

    qs = normalize_questions(questions)
    owner.decision_calls.append({"input": input, "questions": qs})
    raw = produce(input, qs)
    if asyncio.iscoroutine(raw):
        raw = await raw
    result = _coerce_decision(raw, qs, owner.model)
    record_run_tokens(result.usage)
    tracer = get_tracer("fastaiagent.testing.models")
    from opentelemetry.trace import SpanKind

    from fastaiagent.llm.client import decision_span_name, stamp_decision_semconv

    with tracer.start_as_current_span(
        decision_span_name("test", owner.model), kind=SpanKind.CLIENT
    ) as span:
        set_genai_attributes(span, system="test", model=owner.model)
        stamp_decision_semconv(span, provider="test", model=owner.model)
        set_fastaiagent_attributes(
            span,
            **{
                "decision.input": summarize_decision_input(input),
                "decision.questions": _serialize_for_span([q.to_wire() for q in qs]),
            },
        )
        LLMClient._stamp_decision(span, result)
    return result


class TestModel(LLMClient):
    # Tell pytest not to try to collect this class — its name starts with
    # "Test" but it's a runtime helper, not a test class.
    __test__ = False

    """Deterministic ``LLMClient`` that returns canned responses.

    Args:
        response: Either a single string (one response on every call) or a
            list of strings (round-robin through them).
        tool_calls: Canned tool calls returned alongside the *first* canned
            response. Each item is a dict like
            ``{"name": "search", "arguments": {...}}``.
        usage: ``(prompt_tokens, completion_tokens)``.
        model: Model name reported in trace spans (default ``"test-model"``).
        delay_ms: Optional artificial latency before each call.
        decisions: Canned Decisions API results for :meth:`adecide` /
            :meth:`decide` — one ``DecisionResult`` (or its dict form
            ``{"answers": [...]}``) for every call, or a list served in order
            (the last repeats). Without it ``decide()`` raises, so a test never
            gets an answer nobody wrote.

    Example:

        # Single canned response
        TestModel(response="hello")

        # A canned decision
        TestModel(decisions={"answers": [{"type": "predicate", "name": "spam",
                                          "probability": 0.97}]})

        # Round-robin of three responses
        TestModel(response=["one", "two", "three"])

        # Tool call then final answer (paired with FunctionModel for
        # multi-turn flows is also fine).
        TestModel(
            response="Used the tool. Final answer: 42",
            tool_calls=[{"name": "search", "arguments": {"q": "x"}}],
        )
    """

    def __init__(
        self,
        response: str | list[str] | None = "ok",
        *,
        tool_calls: list[dict[str, Any]] | None = None,
        usage: tuple[int, int] = (0, 0),
        model: str = "test-model",
        delay_ms: int = 0,
        decisions: DecisionsArg = None,
    ) -> None:
        super().__init__(provider="test", model=model, api_key="not-used")
        self._turns = _coerce_responses(response, tool_calls, usage)
        self._call_count = 0
        self._delay_ms = max(0, int(delay_ms))
        if decisions is None:
            self._decisions: list[Any] = []
        elif isinstance(decisions, list):
            self._decisions = list(decisions)
        else:
            self._decisions = [decisions]
        self._decision_count = 0
        # Public history for tests to assert against
        self.calls: list[dict[str, Any]] = []
        self.decision_calls: list[dict[str, Any]] = []

    async def adecide(
        self,
        input: Any,
        questions: Any,
        *,
        safety_identifier: str | None = None,
    ) -> DecisionResult:
        if self._delay_ms:
            await asyncio.sleep(self._delay_ms / 1000.0)

        def _next(_input: Any, _qs: list[Any]) -> Any:
            if not self._decisions:
                raise ValueError(
                    "TestModel.decide() has no canned decision; pass "
                    "TestModel(decisions={'answers': [...]})"
                )
            idx = min(self._decision_count, len(self._decisions) - 1)
            self._decision_count += 1
            return self._decisions[idx]

        return await _test_decide(self, input, questions, _next)

    def _next_turn(self) -> _CannedTurn:
        if self._call_count < len(self._turns):
            turn = self._turns[self._call_count]
        else:
            turn = self._turns[-1]
        self._call_count += 1
        return turn

    async def acomplete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if self._delay_ms:
            await asyncio.sleep(self._delay_ms / 1000.0)
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        turn = self._next_turn()
        prompt_t, comp_t = turn.usage
        finish = "tool_calls" if turn.tool_calls else "stop"
        response = LLMResponse(
            content=turn.text or None,
            tool_calls=list(turn.tool_calls),
            usage={
                "prompt_tokens": prompt_t,
                "completion_tokens": comp_t,
                "total_tokens": prompt_t + comp_t,
            },
            model=self.model,
            finish_reason=finish,
            latency_ms=self._delay_ms,
        )
        _record_test_completion(model=self.model, messages=messages, tools=tools, response=response)
        return response

    async def astream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        if self._delay_ms:
            await asyncio.sleep(self._delay_ms / 1000.0)
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        turn = self._next_turn()
        # Build a minimal LLMResponse for the trace span.
        prompt_t, comp_t = turn.usage
        finish = "tool_calls" if turn.tool_calls else "stop"
        response = LLMResponse(
            content=turn.text or None,
            tool_calls=list(turn.tool_calls),
            usage={
                "prompt_tokens": prompt_t,
                "completion_tokens": comp_t,
                "total_tokens": prompt_t + comp_t,
            },
            model=self.model,
            finish_reason=finish,
        )
        _record_test_completion(model=self.model, messages=messages, tools=tools, response=response)

        if turn.text:
            yield TextDelta(text=turn.text)
        for tc in turn.tool_calls:
            yield ToolCallStart(call_id=tc.id, tool_name=tc.name)
            yield ToolCallEnd(call_id=tc.id, tool_name=tc.name, arguments=dict(tc.arguments))
        yield Usage(prompt_tokens=prompt_t, completion_tokens=comp_t)
        yield StreamDone()


# A FunctionModel responder may return any of:
#   - "text answer"
#   - ("text answer", [{"name": "tool", "arguments": {...}}])
#   - ("", [{"name": "tool", "arguments": {...}}])     # tool-call only
#   - LLMResponse(...)                                 # full control
# Responders may be sync or async. The wrapper awaits any coroutine
# returned, so both forms are first-class.
ResponderReturn = (
    str
    | tuple[str, list[dict[str, Any]]]
    | tuple[str, list[ToolCall]]
    | LLMResponse
)
ResponderFn = Callable[[list[Message]], ResponderReturn | Awaitable[ResponderReturn]]


def _normalise_responder_return(
    value: ResponderReturn, *, model: str, base_index: int
) -> LLMResponse:
    """Coerce whatever the user returned into a normalised LLMResponse."""
    if isinstance(value, LLMResponse):
        return value
    if isinstance(value, str):
        return LLMResponse(
            content=value or None,
            tool_calls=[],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model=model,
            finish_reason="stop" if value else "stop",
        )
    if isinstance(value, tuple) and len(value) == 2:
        text, tool_calls = value
        tcs = _coerce_tool_calls(tool_calls, base_index=base_index)
        return LLMResponse(
            content=text or None,
            tool_calls=tcs,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model=model,
            finish_reason="tool_calls" if tcs else "stop",
        )
    raise TypeError(
        "FunctionModel responder must return str, "
        "(str, list[tool_call_dict | ToolCall]), or LLMResponse — got "
        f"{type(value).__name__}"
    )


class FunctionModel(LLMClient):
    """Wrap a callable as an LLMClient.

    The callable receives the conversation as ``list[Message]`` and returns
    one of:
        - ``str`` (final answer text)
        - ``(text, list[tool_call_dict | ToolCall])`` (mixed)
        - ``LLMResponse`` (full control over the wire shape)

    Useful for state machines: track call count in a closure to return one
    response on the first invocation and another on the second.

    Example:

        def responder(messages):
            user = messages[-1].content if messages else ""
            if "weather" in str(user):
                return "", [{"name": "get_weather", "arguments": {"city": "Paris"}}]
            return "It is sunny."

        agent = Agent(name="weather-bot",
                      llm=FunctionModel(responder),
                      tools=[get_weather])
    """

    def __init__(
        self,
        fn: ResponderFn,
        *,
        model: str = "function-model",
        delay_ms: int = 0,
        decide_fn: Callable[[Any, list[Any]], Any] | None = None,
    ) -> None:
        super().__init__(provider="test", model=model, api_key="not-used")
        if not callable(fn):
            raise TypeError("FunctionModel requires a callable")
        if decide_fn is not None and not callable(decide_fn):
            raise TypeError("FunctionModel decide_fn must be callable")
        self._fn = fn
        self._decide_fn = decide_fn
        self._call_count = 0
        self._delay_ms = max(0, int(delay_ms))
        self.calls: list[dict[str, Any]] = []
        self.decision_calls: list[dict[str, Any]] = []

    async def adecide(
        self,
        input: Any,
        questions: Any,
        *,
        safety_identifier: str | None = None,
    ) -> DecisionResult:
        """Answer from ``decide_fn(input, questions)`` (sync or async).

        It returns a ``DecisionResult`` or its dict form ``{"answers": [...]}``.
        """
        if self._decide_fn is None:
            raise ValueError("FunctionModel.decide() needs FunctionModel(..., decide_fn=...)")
        if self._delay_ms:
            await asyncio.sleep(self._delay_ms / 1000.0)
        return await _test_decide(self, input, questions, self._decide_fn)

    async def acomplete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if self._delay_ms:
            await asyncio.sleep(self._delay_ms / 1000.0)
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        start = time.monotonic()
        # Allow async responders too — common when reusing real test
        # fixtures that already await on something.
        raw: Any = self._fn(messages)
        if asyncio.iscoroutine(raw):
            raw = await raw
        response = _normalise_responder_return(raw, model=self.model, base_index=self._call_count)
        self._call_count += 1
        response.latency_ms = int((time.monotonic() - start) * 1000)
        _record_test_completion(model=self.model, messages=messages, tools=tools, response=response)
        return response

    async def astream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        # Streaming path: reuse acomplete and replay as events. This is the
        # same pattern MockLLMClient uses in tests/conftest.py.
        response = await self.acomplete(messages, tools, **kwargs)
        if response.content:
            yield TextDelta(text=response.content)
        for tc in response.tool_calls:
            yield ToolCallStart(call_id=tc.id, tool_name=tc.name)
            yield ToolCallEnd(call_id=tc.id, tool_name=tc.name, arguments=dict(tc.arguments))
        yield Usage(
            prompt_tokens=int(response.usage.get("prompt_tokens", 0)),
            completion_tokens=int(response.usage.get("completion_tokens", 0)),
        )
        yield StreamDone()


__all__ = ["TestModel", "FunctionModel"]
