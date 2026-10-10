"""Proof for "3 · One candidate, one agent" on docs/evaluation/autollm-how-it-works.md.

Offline: the SDK's own FunctionModel, no API key.

A candidate with a new prompt AND few-shot demos is applied to an agent that has
no memory. The original is untouched, the demos reach the prompt, and the second
customer's prompt carries nothing of the first customer's ticket.
"""

import fastaiagent as fa
from fastaiagent.optimize import Candidate, apply_candidate
from fastaiagent.testing import FunctionModel

calls: list[dict[str, list[str]]] = []


def model(messages):
    calls.append(
        {
            "system": [m.content for m in messages if m.role.value == "system"],
            "user": [m.content for m in messages if m.role.value == "user"],
        }
    )
    return '{"queue": "billing", "priority": "P1"}'


base = fa.Agent(name="triage", system_prompt="You triage tickets.", llm=FunctionModel(model))
demos = [{"input": "charged EUR 900 twice", "output": '{"queue": "billing", "priority": "P1"}'}]
tuned = apply_candidate(
    base, Candidate(system_prompt="You triage tickets by the house rules.", fewshot_demos=demos)
)

tuned.run("customer one: my card is 4111 1111 1111 1111")
tuned.run("customer two: hello")

print("original prompt :", repr(base.system_prompt))
print("original memory :", base.memory)
print("tuned prompt    :", repr(tuned.system_prompt))
print("demo in prompt  :", any("charged EUR 900 twice" in s for s in calls[-1]["system"]))
print("customer two saw:", calls[-1]["user"])
