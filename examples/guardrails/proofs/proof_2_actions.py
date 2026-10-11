"""Proof for "2 · Between a failure and what it costs" on docs/guardrails/guardrail-boundaries.md.

Offline: a real Agent over the SDK's own TestModel and FunctionModel, real regex rules.

A failure has a cost, and the cost is a third axis: block, warn, mask, override or
reask. A mask rewrites the payload for everything downstream — the next rule and
the model. Where a rewrite cannot be applied, the run blocks instead.
"""

import asyncio

import _common  # noqa: F401
from _common import firings, heading

from fastaiagent import Agent
from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.guardrail import Guardrail, GuardrailPosition, GuardrailType
from fastaiagent.testing import FunctionModel, TestModel

SSN = r"\b\d{3}-\d{2}-\d{4}\b"
REPLY = "The customer's SSN is 123-45-6789."


def ssn_rule(position: GuardrailPosition, action: str, *, name: str = "ssn", **config) -> Guardrail:
    return Guardrail(name=name, guardrail_type=GuardrailType.regex, position=position,
                     config={"pattern": SSN, **config}, action=action)


heading("the same failure, five costs (output position)")
for action in ("block", "warn", "mask", "override"):
    cfg = {"override_message": "I can't share that."} if action == "override" else {}
    agent = Agent(name="t", llm=TestModel(response=REPLY), guardrails=[ssn_rule(GuardrailPosition.output, action, **cfg)])
    try:
        r = agent.run("go")
        print(f"{action:<9} → output={r.output!r}  firing={firings(r)[0]}")
    except GuardrailBlockedError as e:
        print(f"{action:<9} → GuardrailBlockedError: {e}")

calls: list[str] = []


def responder(messages):
    calls.append(messages[-1].content)
    return REPLY if len(calls) == 1 else "The customer's SSN is on file."


agent = Agent(name="t", llm=FunctionModel(responder), guardrails=[ssn_rule(GuardrailPosition.output, "reask")])
r = agent.run("go")
print(f"{'reask':<9} → output={r.output!r}  model calls={len(calls)}")
print(f"{'':<9}   second call's last message: {calls[1][:90]!r}…")

heading("a mask at input: what the model is sent")
got: list[str] = []
agent = Agent(name="t", llm=FunctionModel(lambda ms: got.append(ms[-1].content) or "noted"),
              guardrails=[ssn_rule(GuardrailPosition.input, "mask")])
agent.run("My SSN is 123-45-6789, please update it.")
print("the model received:", repr(got[0]))

heading("a mask feeds the next rule")
agent = Agent(name="t", llm=TestModel(response=REPLY), guardrails=[
    ssn_rule(GuardrailPosition.output, "mask", name="mask-ssn"),
    ssn_rule(GuardrailPosition.output, "block", name="block-ssn"),
])
r = agent.run("go")
print("output :", repr(r.output))
print("firings:", firings(r), "— block-ssn judged the masked text")

heading("an observer's rewrite is evidence, never applied")
watch = Guardrail(name="watch-ssn", guardrail_type=GuardrailType.regex, position=GuardrailPosition.output,
                  config={"pattern": SSN}, action="mask", blocking=False)
r = Agent(name="t", llm=TestModel(response=REPLY), guardrails=[watch]).run("go")
print("output :", repr(r.output), "· firing:", firings(r)[0])

heading("where a rewrite cannot be applied: a streamed reply")


async def stream():
    agent = Agent(name="t", llm=TestModel(response=REPLY), guardrails=[ssn_rule(GuardrailPosition.output, "mask")])
    text = ""
    try:
        async for ev in agent.astream("go"):
            text += getattr(ev, "text", "")
    except GuardrailBlockedError as e:
        return text, str(e)
    return text, None


streamed, err = asyncio.run(stream())
print("already streamed:", repr(streamed))
print("then            :", err)

heading("an errored check blocks whatever the action says")


def down(_):
    raise RuntimeError("detector down")


r = Guardrail(name="pii-mask", fn=down, action="mask", on_error="block").execute("text")
print(f"passed={r.passed} errored={r.errored} action={r.action!r} action_taken={r.action_taken!r}")
