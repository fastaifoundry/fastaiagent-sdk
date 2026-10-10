"""Proof for "5 · Between free text and a typed object" on docs/llm/llm-boundaries.md.

Live for the first half: OPENAI_API_KEY and ANTHROPIC_API_KEY (a missing key skips
that provider). Offline for the second: the SDK's own FunctionModel.

The same output_type on two providers — one with native JSON-schema support, one
without — gives the same parsed object. When a reply doesn't parse, the agent
re-asks with the parse error written as prose.
"""

import os

import _common  # noqa: F401
from _common import heading
from pydantic import BaseModel

from fastaiagent import Agent, LLMClient
from fastaiagent.llm import UserMessage
from fastaiagent.testing import FunctionModel


class City(BaseModel):
    name: str
    country: str


heading("output_type on the client: two providers, one parsed object")
for provider, model, env in (("openai", "gpt-4.1-mini", "OPENAI_API_KEY"),
                             ("anthropic", "claude-sonnet-5-5", "ANTHROPIC_API_KEY")):
    if not os.environ.get(env):
        print(f"{provider:<10} skipped: {env} not set")
        continue
    llm = LLMClient(provider=provider, model=model, max_tokens=200)
    r = llm.complete([UserMessage("Which city has the Eiffel Tower?")], output_type=City)
    print(f"{provider:<10} content={r.content!r}")
    print(f"{'':<10} parsed ={r.parsed!r} ({type(r.parsed).__name__})")

heading("the agent re-asks when a reply does not parse (offline)")
seen: list[str] = []


def responder(messages):
    seen.append(messages[-1].content)
    if len(seen) == 1:
        return "Paris, which is in France."  # not JSON
    return '{"name": "Paris", "country": "France"}'


agent = Agent(name="geo", llm=FunctionModel(responder), output_type=City)
result = agent.run("Which city has the Eiffel Tower?")
print("model calls   :", len(seen))
print("second call's last message:", repr(seen[1][:150]), "…")
print("result.parsed :", result.parsed, "· output:", result.output)
