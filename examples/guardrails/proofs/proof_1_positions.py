"""Proof for "1 · Between the gate and the run" on docs/guardrails/guardrail-boundaries.md.

Offline: a real Agent over the SDK's own TestModel and FunctionModel, real guardrails.

Four positions, one executor. At each position the blocking rules run first and in
order, and the first one that fails stops the run before the observers run. The
observers run after, in parallel; one that crashes is recorded as a failure, never a
pass, and never stops the run.
"""

import _common  # noqa: F401
from _common import firings, heading

from fastaiagent import Agent, FunctionTool
from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.guardrail import Guardrail, GuardrailPosition, GuardrailResult
from fastaiagent.testing import FunctionModel, TestModel

order: list[str] = []


def rule(name: str, position: GuardrailPosition, *, blocking: bool = True, verdict: str = "pass") -> Guardrail:
    def fn(data):
        order.append(f"{position.value}:{name}")
        if verdict == "raise":
            raise RuntimeError("judge unreachable")
        return GuardrailResult(passed=(verdict == "pass"), message=f"{name}: {verdict}")

    return Guardrail(name=name, position=position, blocking=blocking, fn=fn)


heading("one gate, two observers")
agent = Agent(
    name="t",
    llm=TestModel(response="ok"),
    guardrails=[
        rule("gate", GuardrailPosition.input),
        rule("watch-a", GuardrailPosition.input, blocking=False, verdict="fail"),
        rule("watch-b", GuardrailPosition.input, blocking=False, verdict="raise"),
    ],
)
result = agent.run("hello")
print("ran, in order :", order)
print("run status    :", result.status, "· output:", repr(result.output))
print("firings       :", firings(result))

heading("the gate fails: the observers never run")
order.clear()
agent = Agent(
    name="t",
    llm=TestModel(response="ok"),
    guardrails=[
        rule("gate", GuardrailPosition.input, verdict="fail"),
        rule("watch-a", GuardrailPosition.input, blocking=False),
    ],
)
try:
    agent.run("hello")
except GuardrailBlockedError as e:
    print("raised        :", f"GuardrailBlockedError({e.guardrail_name!r}): {e}")
print("ran, in order :", order)

heading("four positions, one tool call")
order.clear()
calls = {"n": 0}


def responder(messages):
    calls["n"] += 1
    if calls["n"] == 1:
        return "", [{"name": "lookup", "arguments": {"order": "1042"}}]
    return "Order 1042 shipped."


def lookup(order: str) -> str:
    """Look an order up."""
    return f"order {order}: shipped"


seen: dict[str, str] = {}


def probe(position: GuardrailPosition) -> Guardrail:
    def fn(data):
        order.append(position.value)
        seen[position.value] = f"{type(data).__name__}: {str(data)[:48]!r}"
        return GuardrailResult(passed=True)

    return Guardrail(name=f"probe-{position.value}", position=position, fn=fn)


agent = Agent(
    name="t",
    llm=FunctionModel(responder),
    tools=[FunctionTool(name="lookup", fn=lookup)],
    guardrails=[probe(p) for p in GuardrailPosition],
)
agent.run("Where is order 1042?")
print("ran, in order :", order)
for position, what in seen.items():
    print(f"  {position:<12} judged {what}")
