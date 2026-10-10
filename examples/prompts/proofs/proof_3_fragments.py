"""Proof for "3 · Between a fragment and the prompts that use it" on docs/prompts/prompt-boundaries.md.

Offline: a registry on a scratch local.db; the SDK's own TestModel.

A fragment is one row, updated in place. Change it, and every prompt that uses it
changes on its next load — same version number, nothing in diff(). The trace is
the only record of the text the model was actually sent.
"""

import json

import _common
from _common import SpanCollector, heading

from fastaiagent import Agent
from fastaiagent.prompt import PromptRegistry
from fastaiagent.testing import TestModel

col = SpanCollector()
reg = PromptRegistry(path=str(_common.SCRATCH))

reg.register_fragment("tone", "Be formal.")
reg.register("support", "You help customers. {{@tone}}")  # v1
before = reg.load("support")
reg.register_fragment("tone", "Be casual and brief.")  # the same fragment, edited
after = reg.load("support")

heading("one prompt version, two different texts")
print(f"before the edit: v{before.version}  {before.template!r}")
print(f"after the edit : v{after.version}  {after.template!r}")
print("diff(1, 1)     :", reg.diff("support", 1, 1).splitlines()[-1].strip())
print("list()         :", reg.list())

heading("what each run's trace recorded")
for prompt in (before, after):
    Agent(name="support", system_prompt=prompt, llm=TestModel(response="ok")).run("hi")
for attrs in col.llm():
    messages = json.loads(attrs["gen_ai.request.messages"])
    system = [m["content"] for m in messages if m["role"] == "system"]
    print(f"prompt.version={attrs['fastaiagent.prompt.version']}  system prompt sent: {system}")
