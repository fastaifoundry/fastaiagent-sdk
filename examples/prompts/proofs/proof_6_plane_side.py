"""Proof for "6 · Between your laptop and the plane" on docs/prompts/prompt-boundaries.md.

Offline: two scratch registries, a child process, the SDK's own TestModel. No
control plane is contacted; this shows what the SDK does on its side of that line.
"""

import subprocess
import sys

import _common
from _common import SpanCollector, heading

from fastaiagent import Agent
from fastaiagent._internal.errors import PlatformNotConnectedError, PromptNotFoundError
from fastaiagent.prompt import Prompt, PromptRegistry
from fastaiagent.testing import TestModel

col = SpanCollector()
a_dir, b_dir = _common.SCRATCH / "laptop-a", _common.SCRATCH / "laptop-b"
reg_a = PromptRegistry(path=str(a_dir))
reg_a.register("support-prompt", "You are the support agent for {{company}}.")

heading("a local registry is rows in one local.db")
print("registry A holds          :", [p["name"] for p in reg_a.list()])
try:
    PromptRegistry(path=str(b_dir)).load("support-prompt")
except PromptNotFoundError as e:
    print("registry B, another folder:", e)
child = subprocess.run(
    [sys.executable, "-c",
     "from fastaiagent.prompt import PromptRegistry;"
     f"p = PromptRegistry(path={str(a_dir)!r}).load('support-prompt');"
     "print(f'v{p.version} {p.template!r}')"],
    capture_output=True, text=True, check=True,
)
print("a new process, A's folder :", child.stdout.strip())

heading("not connected: get() is local, the plane paths refuse")
print("get(slug)                 :", repr(reg_a.get("support-prompt").template), "· source =",
      reg_a.get("support-prompt").source)
for label, call in (
    ("get(source='platform')", lambda: reg_a.get("support-prompt", source="platform")),
    ("publish(...)", lambda: reg_a.publish("support-prompt", "text", [])),
):
    try:
        call()
    except PlatformNotConnectedError as e:
        print(f"{label:<26}: PlatformNotConnectedError: {e}")

heading("what a pushed agent sends")
local = Agent(name="support-bot", system_prompt=reg_a.load("support-prompt"), llm=TestModel())
d = local.to_dict()
print("local Prompt   → prompt_slug:", d.get("prompt_slug"), "· system_prompt:", repr(d["system_prompt"]))
linked = Agent(name="support-bot", prompt_slug="support-prompt", llm=TestModel())
d = linked.to_dict()
print("prompt_slug=   → prompt_slug:", d.get("prompt_slug"), "· system_prompt:", repr(d["system_prompt"]))

heading("a plane prompt, as the registry builds one from the plane's reply")
plane_prompt = Prompt(name="support-prompt", template="You are the support agent for Acme.",
                      version=3, slug="support-prompt", source="platform", environment="production")
agent = Agent(name="support-bot", system_prompt=plane_prompt, llm=TestModel(response="ok"))
agent.run("hi")
print("to_dict() prompt_slug     :", agent.to_dict().get("prompt_slug"), "(auto-linked)")
attrs = col.llm()[-1]
print("llm span                  :", {k.removeprefix("fastaiagent."): v for k, v in attrs.items()
                                    if k.startswith("fastaiagent.prompt.")})
