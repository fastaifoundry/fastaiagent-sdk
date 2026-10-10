"""Proof for "2 · Between the reply and your code" on docs/llm/llm-boundaries.md.

Live: needs OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY and GROQ_API_KEY.
A provider whose key is missing is skipped with a line saying so.

The same tool question to four providers on three wires. Every reply comes back
as one LLMResponse: the same finish_reason vocabulary, tool-call arguments as a
dict, and prompt_tokens / completion_tokens in usage.
"""

import os

import _common  # noqa: F401
from _common import heading

from fastaiagent import LLMClient
from fastaiagent.llm import UserMessage

PROVIDERS = [
    ("openai", "gpt-4.1-mini", "OPENAI_API_KEY"),
    ("anthropic", "claude-sonnet-5-5", "ANTHROPIC_API_KEY"),
    ("gemini", "gemini-2.5-flash", "GEMINI_API_KEY"),
    ("groq", "openai/gpt-oss-120b", "GROQ_API_KEY"),
]
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}}]

heading("a plain reply")
print(f"{'provider':<10} {'finish':<8} {'content':<10} {'usage keys (normalized)':<44} latency")
for provider, model, env in PROVIDERS:
    if not os.environ.get(env):
        print(f"{provider:<10} skipped: {env} not set")
        continue
    llm = LLMClient(provider=provider, model=model, max_tokens=400)
    r = llm.complete([UserMessage("Reply with the single word PONG.")])
    keys = [k for k in ("prompt_tokens", "completion_tokens", "total_tokens") if k in r.usage]
    extra = len(r.usage) - len(keys)
    print(f"{provider:<10} {r.finish_reason:<8} {str(r.content).strip()[:10]:<10} "
          f"{str(keys) + (f' +{extra} provider-specific' if extra else ''):<44} {r.latency_ms} ms")

heading("a tool call")
print(f"{'provider':<10} {'finish':<11} {'tool':<12} {'arguments':<20} {'type':<6} content")
for provider, model, env in PROVIDERS:
    if not os.environ.get(env):
        continue
    llm = LLMClient(provider=provider, model=model, max_tokens=400)
    r = llm.complete([UserMessage("What is the weather in Paris? Use the tool.")], tools=TOOLS)
    tc = r.tool_calls[0] if r.tool_calls else None
    print(f"{provider:<10} {r.finish_reason:<11} {tc.name if tc else '-':<12} "
          f"{str(tc.arguments) if tc else '-':<20} {type(tc.arguments).__name__ if tc else '-':<6} {r.content!r}")
