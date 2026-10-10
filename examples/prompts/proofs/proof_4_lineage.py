"""Proof for "4 · Between the prompt and the run" on docs/prompts/prompt-boundaries.md.

Offline: a registry on a scratch local.db; the SDK's own TestModel and FunctionModel.

Pass the Prompt object and every llm span the agent makes is stamped with the
prompt's name and version — run() and astream() alike. Pass the formatted string
and nothing is stamped. An agent run inside another stamps its own prompt.
"""

import asyncio

import _common
from _common import SpanCollector, heading

from fastaiagent import Agent, FunctionTool
from fastaiagent.prompt import PromptRegistry
from fastaiagent.testing import FunctionModel, TestModel

col = SpanCollector()
reg = PromptRegistry(path=str(_common.SCRATCH))
reg.register("triage", "You triage tickets. Reply with the queue name.")
reg.register("triage", "You triage tickets. Reply with the queue name only.")  # v2
reg.register("summarizer", "Summarize the ticket in one line.")

PROMPT_KEYS = ("fastaiagent.prompt.name", "fastaiagent.prompt.version")


def stamps() -> list[str]:
    out = []
    for attrs in col.llm():
        name, version = (attrs.get(k) for k in PROMPT_KEYS)
        out.append(f"{name} v{version}" if name else "(no prompt)")
    col.reset()
    return out


prompt = reg.load("triage")  # latest: v2
agent = Agent(name="triage", system_prompt=prompt, llm=TestModel(response="billing"))

heading("the Prompt object: every path is stamped")
agent.run("I was charged twice")
print("run()    :", stamps())


async def _stream() -> None:
    async for _ in agent.astream("Refund please"):
        pass


asyncio.run(_stream())
print("astream():", stamps())

heading("the formatted string: nothing to stamp")
Agent(name="triage", system_prompt=prompt.format(), llm=TestModel(response="billing")).run("hi")
print("run()    :", stamps())

heading("an agent inside a tool stamps its own prompt")
inner = Agent(name="summarizer", system_prompt=reg.load("summarizer"), llm=TestModel(response="a refund"))


def summarize(text: str) -> str:
    """Summarize a ticket."""
    return inner.run(text).output


calls = {"n": 0}


def responder(messages):
    calls["n"] += 1
    if calls["n"] == 1:
        return "", [{"name": "summarize", "arguments": {"text": "I was charged twice"}}]
    return "billing"


outer = Agent(
    name="triage",
    system_prompt=prompt,
    llm=FunctionModel(responder),
    tools=[FunctionTool(name="summarize", fn=summarize)],
)
outer.run("I was charged twice")
print("llm spans, in order:", stamps())
