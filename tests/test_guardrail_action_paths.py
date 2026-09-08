"""Where a rewritten payload actually lands — the four guardrail call sites.

``mask`` and ``override`` change the payload, so ``execute_guardrails`` can no
longer be a function that either returns verdicts or raises. These tests drive a
real ``Agent`` over the offline ``TestModel`` at each of the four positions and
assert both halves of the contract:

* where the rewrite **can** be applied faithfully, it is — and the caller gets
  the rewritten value, not the original;
* where it **cannot**, the run blocks. Passing the payload through untouched
  would defeat the control, so every one of those paths fails closed.

No mocks: a real agent, real guardrails, real tools.
"""

from __future__ import annotations

import pytest

from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.agent.agent import Agent, AgentConfig
from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition, GuardrailType
from fastaiagent.multimodal import Image
from fastaiagent.testing.models import FunctionModel, TestModel
from fastaiagent.tool import tool

SSN = r"\b\d{3}-\d{2}-\d{4}\b"


def _mask_rule(position: GuardrailPosition, *, name: str = "mask-ssn", **config) -> Guardrail:
    return Guardrail(
        name=name,
        guardrail_type=GuardrailType.regex,
        position=position,
        config={"pattern": SSN, "should_match": False, **config},
        action="mask",
    )


# --------------------------------------------------------------------------- #
# output — the position that can do everything
# --------------------------------------------------------------------------- #
def test_a_masked_output_is_what_the_caller_receives() -> None:
    agent = Agent(
        name="t",
        llm=TestModel(response="his ssn is 123-45-6789"),
        guardrails=[_mask_rule(GuardrailPosition.output)],
    )
    result = agent.run("go")
    assert result.output == "his ssn is [REDACTED]"


def test_an_overridden_output_is_replaced_with_the_operators_copy() -> None:
    rule = Guardrail(
        name="refuse",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.output,
        config={"pattern": SSN, "override_message": "I can't share that."},
        action="override",
    )
    agent = Agent(name="t", llm=TestModel(response="his ssn is 123-45-6789"), guardrails=[rule])
    assert agent.run("go").output == "I can't share that."


def test_a_rewritten_output_is_re_parsed_rather_than_left_stale() -> None:
    """``parsed`` was derived from the text the model produced. Once a guardrail
    rewrites that text, the old parse describes something the caller never sees."""

    class Reply(__import__("pydantic").BaseModel):  # noqa: N806
        note: str

    rule = _mask_rule(GuardrailPosition.output)
    agent = Agent(
        name="t",
        llm=TestModel(response='{"note": "ssn 123-45-6789"}'),
        guardrails=[rule],
        output_type=Reply,
    )
    result = agent.run("go")
    assert "123-45-6789" not in result.output
    assert result.parsed is not None
    assert result.parsed.note == "ssn [REDACTED]"


def test_a_warn_output_rule_lets_the_reply_through_untouched() -> None:
    rule = Guardrail(
        name="watch",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.output,
        config={"pattern": SSN},
        action="warn",
    )
    agent = Agent(name="t", llm=TestModel(response="his ssn is 123-45-6789"), guardrails=[rule])
    assert agent.run("go").output == "his ssn is 123-45-6789"


# --------------------------------------------------------------------------- #
# reask — the one action the plane cannot perform
# --------------------------------------------------------------------------- #
def test_reask_re_prompts_the_model_and_accepts_the_corrected_reply() -> None:
    rule = Guardrail(
        name="no-ssn",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.output,
        config={"pattern": SSN},
        action="reask",
    )
    agent = Agent(
        name="t",
        # First reply trips the rule, second one doesn't.
        llm=TestModel(response=["his ssn is 123-45-6789", "I won't share that."]),
        guardrails=[rule],
    )
    assert agent.run("go").output == "I won't share that."


def test_reask_blocks_once_the_retry_cap_is_exhausted() -> None:
    """A re-ask that never converges must not become a silent pass."""
    rule = Guardrail(
        name="no-ssn",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.output,
        config={"pattern": SSN},
        action="reask",
    )
    agent = Agent(
        name="t",
        llm=TestModel(response="his ssn is 123-45-6789"),  # never improves
        guardrails=[rule],
    )
    with pytest.raises(GuardrailBlockedError) as exc:
        agent.run("go")
    assert "re-asked" in str(exc.value)


def test_guardrail_retries_zero_makes_a_reask_rule_block_outright() -> None:
    rule = Guardrail(
        name="no-ssn",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.output,
        config={"pattern": SSN},
        action="reask",
    )
    agent = Agent(
        name="t",
        llm=TestModel(response=["his ssn is 123-45-6789", "clean"]),
        guardrails=[rule],
        config=AgentConfig(guardrail_retries=0),
    )
    with pytest.raises(GuardrailBlockedError):
        agent.run("go")


def test_the_re_ask_turn_is_counted_in_the_token_total() -> None:
    rule = Guardrail(
        name="no-ssn",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.output,
        config={"pattern": SSN},
        action="reask",
    )
    agent = Agent(
        name="t",
        llm=TestModel(response=["his ssn is 123-45-6789", "clean"], usage=(5, 5)),
        guardrails=[rule],
    )
    result = agent.run("go")
    assert result.output == "clean"
    assert result.tokens_used >= 20, "both turns should be billed, not just the first"


# --------------------------------------------------------------------------- #
# input — plain text rewrites, multimodal blocks
# --------------------------------------------------------------------------- #
def test_a_masked_input_is_what_reaches_the_model() -> None:
    captured: list[str] = []

    def _record(messages):
        captured.append(str(messages[-1].content))
        return "ok"

    agent = Agent(
        name="t",
        llm=FunctionModel(_record),
        guardrails=[_mask_rule(GuardrailPosition.input)],
    )
    agent.run("my ssn is 123-45-6789")
    assert captured, "the model was never called"
    assert "123-45-6789" not in captured[0]
    assert "[REDACTED]" in captured[0]


def test_a_rewritten_multimodal_input_blocks_rather_than_dropping_the_parts() -> None:
    """The judged text is a *summary* of the parts. Substituting it would drop
    the image; passing the parts through would hand the model the text the rule
    redacted. Neither is acceptable."""
    agent = Agent(
        name="t",
        llm=TestModel(response="ok"),
        guardrails=[_mask_rule(GuardrailPosition.input)],
    )
    png = Image.from_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32, media_type="image/png")
    with pytest.raises(GuardrailBlockedError) as exc:
        agent.run(["my ssn is 123-45-6789", png])
    assert "multimodal" in str(exc.value)


# --------------------------------------------------------------------------- #
# streaming — the text has already left the building
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_a_streamed_output_cannot_be_masked_after_the_fact() -> None:
    agent = Agent(
        name="t",
        llm=TestModel(response="his ssn is 123-45-6789"),
        guardrails=[_mask_rule(GuardrailPosition.output)],
    )
    with pytest.raises(GuardrailBlockedError) as exc:
        async for _ in agent.astream("go"):
            pass
    assert "already been streamed" in str(exc.value)


@pytest.mark.asyncio
async def test_a_streamed_output_cannot_be_re_asked_either() -> None:
    rule = Guardrail(
        name="no-ssn",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.output,
        config={"pattern": SSN},
        action="reask",
    )
    agent = Agent(name="t", llm=TestModel(response="his ssn is 123-45-6789"), guardrails=[rule])
    with pytest.raises(GuardrailBlockedError) as exc:
        async for _ in agent.astream("go"):
            pass
    assert "re-ask" in str(exc.value)


@pytest.mark.asyncio
async def test_a_warn_rule_still_streams_normally() -> None:
    rule = Guardrail(
        name="watch",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.output,
        config={"pattern": SSN},
        action="warn",
    )
    agent = Agent(name="t", llm=TestModel(response="his ssn is 123-45-6789"), guardrails=[rule])
    events = [e async for e in agent.astream("go")]
    assert events, "the stream should complete"


# --------------------------------------------------------------------------- #
# tool_call / tool_result
# --------------------------------------------------------------------------- #
@tool()
def echo(text: str) -> str:
    """Echo the text back."""
    return text


@tool()
def leak(_q: str) -> str:
    """Return something that trips the rule."""
    return "the ssn is 123-45-6789"


def _call_then_answer(name: str, arguments: dict) -> FunctionModel:
    """One tool call on the first turn, a plain answer on every turn after."""
    state = {"called": False}

    def _responder(_messages):
        if state["called"]:
            return "done"
        state["called"] = True
        return "", [{"name": name, "arguments": arguments}]

    return FunctionModel(_responder)


def test_a_masked_tool_call_reaches_the_tool_with_redacted_arguments() -> None:
    seen: list[str] = []

    @tool()
    def record(text: str) -> str:
        """Record what actually arrived."""
        seen.append(text)
        return "done"

    agent = Agent(
        name="t",
        llm=_call_then_answer("record", {"text": "ssn 123-45-6789"}),
        tools=[record],
        guardrails=[_mask_rule(GuardrailPosition.tool_call)],
    )
    agent.run("go")
    assert seen == ["ssn [REDACTED]"]


def test_an_overridden_tool_call_blocks_because_prose_is_not_an_arguments_object() -> None:
    rule = Guardrail(
        name="refuse-call",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.tool_call,
        config={"pattern": SSN, "override_message": "nope"},
        action="override",
    )
    agent = Agent(
        name="t",
        llm=_call_then_answer("echo", {"text": "ssn 123-45-6789"}),
        tools=[echo],
        guardrails=[rule],
    )
    with pytest.raises(GuardrailBlockedError) as exc:
        agent.run("go")
    assert "arguments object" in str(exc.value)


def test_a_masked_tool_result_is_what_the_model_is_told() -> None:
    agent = Agent(
        name="t",
        llm=_call_then_answer("leak", {"_q": "x"}),
        tools=[leak],
        guardrails=[_mask_rule(GuardrailPosition.tool_result)],
    )
    result = agent.run("go")
    assert result.tool_calls
    assert "123-45-6789" not in str(result.tool_calls)
    assert "[REDACTED]" in str(result.tool_calls)
