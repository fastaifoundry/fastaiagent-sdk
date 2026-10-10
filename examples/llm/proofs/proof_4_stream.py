"""Proof for "4 · Between one reply and a stream of it" on docs/llm/llm-boundaries.md.

Live: needs OPENAI_API_KEY (gpt-4.1-mini). The agent half runs offline on the
SDK's own FunctionModel.

The same call, two shapes: astream() yields typed events, stream() folds them
back into the LLMResponse complete() would return, and both leave the same
llm span with usage on it. Inside an agent, a tool loop looks like one stream.
"""

import asyncio
import os
import sys
from collections import Counter

import _common  # noqa: F401
from _common import SpanCollector, heading

from fastaiagent import Agent, FunctionTool, LLMClient
from fastaiagent.llm import StreamDone, TextDelta, ToolCallEnd, ToolCallStart, Usage, UserMessage
from fastaiagent.testing import FunctionModel

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY is not set — this proof calls a real model.")

col = SpanCollector()
llm = LLMClient(provider="openai", model=os.environ.get("PROOF_MODEL", "gpt-4.1-mini"), temperature=0)
PROMPT = [UserMessage("Count from 1 to 5, separated by commas, nothing else.")]
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}}]

heading("complete() and stream() return the same shape")
a = llm.complete(PROMPT)
b = llm.stream(PROMPT)
for label, r in (("complete()", a), ("stream()  ", b)):
    print(f"{label} content={r.content!r} finish={r.finish_reason!r} usage={r.usage}")


async def events(messages, tools=None):
    seen = []
    async for ev in llm.astream(messages, tools=tools):
        seen.append(ev)
    return seen


heading("astream() is the same call as typed events")
seen = asyncio.run(events(PROMPT))
print("event sequence:", dict(Counter(type(e).__name__ for e in seen)))
print("text joined   :", repr("".join(e.text for e in seen if isinstance(e, TextDelta))))
print("last event    :", type(seen[-1]).__name__, "· usage event:",
      next(e for e in seen if isinstance(e, Usage)))
seen = asyncio.run(events([UserMessage("What is the weather in Paris? Use the tool.")], TOOLS))
tool_events = [e for e in seen if isinstance(e, (ToolCallStart, ToolCallEnd))]
print("with a tool   :", [(type(e).__name__, e.tool_name, getattr(e, 'arguments', None)) for e in tool_events])

heading("both paths leave the same llm span")
for attrs in col.llm()[:2]:
    print({k: v for k, v in attrs.items() if k in ("gen_ai.request.model", "gen_ai.usage.input_tokens",
                                                   "gen_ai.usage.output_tokens", "gen_ai.response.finish_reason")})

heading("inside an agent, a tool loop is one stream (offline, FunctionModel)")
calls = {"n": 0}


def responder(messages):
    calls["n"] += 1
    if calls["n"] == 1:
        return "", [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    return "It is 18°C and sunny in Paris."


def get_weather(city: str) -> str:
    """Current weather for a city."""
    return "18°C and sunny"


agent = Agent(name="weather", llm=FunctionModel(responder), tools=[FunctionTool(name="get_weather", fn=get_weather)])


async def agent_events():
    seen = []
    async for ev in agent.astream("Weather in Paris?"):
        seen.append(ev)
    return seen


seen = asyncio.run(agent_events())
print("model turns:", calls["n"], "· events in order:", [type(e).__name__ for e in seen])
print("StreamDone events inside the agent stream:", sum(isinstance(e, StreamDone) for e in seen))
