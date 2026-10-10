"""Proof for "1 · Between your messages and the wire" on docs/llm/llm-boundaries.md.

Offline: the client is pointed at a local endpoint that records the request and
answers with a canned reply, so the bodies below are the real bytes the SDK
sends; only the model is canned.

One conversation — a system prompt, a question, a tool call and its result — and
one tool definition, rendered for three providers.
"""

import json

import _common  # noqa: F401
from _common import heading
from _recorder import Recorder

from fastaiagent import LLMClient
from fastaiagent.llm import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage

MESSAGES = [
    SystemMessage("You are a weather bot."),
    UserMessage("Weather in Paris?"),
    AssistantMessage(tool_calls=[ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})]),
    ToolMessage("18°C and sunny", tool_call_id="call_1"),
]
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}}]

rec = Recorder()
url = rec.start()
for provider in ("openai", "custom", "anthropic"):
    llm = LLMClient(provider=provider, model="demo", base_url=url, api_key="k", max_tokens=64)
    llm.complete(MESSAGES, tools=TOOLS)
    req = rec.last
    heading(f"provider={provider!r}  →  POST {req['path']}")
    print("headers :", req["headers"])
    body = req["body"]
    print("keys    :", sorted(body))
    print("messages:", json.dumps(body["messages"], ensure_ascii=False))
    if "system" in body:
        print("system  :", repr(body["system"]))
    print("tools   :", json.dumps(body["tools"]))
rec.stop()
