"""An output guardrail does not judge a swarm member's handoff (1.102.0).

A member that hands off ends its turn with the ``__HANDOFF__`` marker, not a sentence for
the person. Output guardrails ran on it all the same, so an allow-style rule such as
``allowed_topics`` read the marker as an off-topic answer and stopped the swarm on its
first handoff. Found by the plane's swarm lab on a Studio-exported swarm (same on 1.88.0
and 1.101.0); a sandbox trial cannot see it, because trials apply no guardrails.

No mocks: the offline ``FunctionModel`` and the built-in keyword ``allowed_topics``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import fastaiagent as fa
from fastaiagent._internal.errors import GuardrailBlockedError, StopAgent
from fastaiagent.agent.middleware import AgentMiddleware
from fastaiagent.guardrail import allowed_topics
from fastaiagent.testing import FunctionModel

ON_TOPIC = ["order", "refund"]


def _swarm(billing_says: str = "Your order 1003 has shipped.") -> fa.Swarm:
    def triage(messages: list[Any]) -> Any:
        return "", [{"name": "handoff_to_billing", "arguments": {"reason": "needs billing"}}]

    def billing(messages: list[Any]) -> Any:
        return billing_says

    def guard() -> list[Any]:
        return [allowed_topics(ON_TOPIC, mode="keyword")]

    return fa.Swarm(
        name="support",
        agents=[
            fa.Agent(name="triage", llm=FunctionModel(triage), guardrails=guard()),
            fa.Agent(name="billing", llm=FunctionModel(billing), guardrails=guard()),
        ],
        entrypoint="triage",
    )


def _run(swarm: fa.Swarm, path: str) -> str:
    if path == "run":
        return swarm.run("Where is order 1003?").output

    async def consume() -> str:
        text = ""
        async for event in swarm.astream("Where is order 1003?"):
            text += getattr(event, "text", "") or ""
        return text

    return asyncio.run(consume())


@pytest.mark.parametrize("path", ["run", "astream"])
def test_a_handoff_is_not_judged_off_topic(path: str) -> None:
    output = _run(_swarm(), path)
    assert "order 1003 has shipped" in output
    assert "__HANDOFF__" not in output


@pytest.mark.parametrize("path", ["run", "astream"])
def test_the_member_that_answers_is_still_checked(path: str) -> None:
    """Skipping the marker must not switch the rule off: the answer is judged."""
    with pytest.raises(GuardrailBlockedError):
        _run(_swarm(billing_says="Cats are wonderful companions."), path)


def test_the_handoff_is_still_recorded_in_the_run() -> None:
    result = _swarm().run("Where is order 1003?")
    assert "handoff_to_billing" in [c["tool_name"] for c in result.tool_calls]


def test_another_stop_is_still_judged_by_the_rules() -> None:
    """Only a handoff is a control signal. Middleware that ends a run with its own
    message ("budget exhausted") is speaking to the person, so the rules still read it."""

    class Halt(AgentMiddleware):
        name = "halt"

        async def before_model(self, ctx: Any, messages: Any) -> Any:
            raise StopAgent("unrelated stop message", reason="budget")

    agent = fa.Agent(
        name="halted",
        llm=FunctionModel(lambda messages: "never reached"),
        middleware=[Halt()],
        guardrails=[allowed_topics(ON_TOPIC, mode="keyword")],
    )
    with pytest.raises(GuardrailBlockedError):
        agent.run("Where is order 1003?")
