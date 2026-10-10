"""Proof for "3 · Between a provider's name and the code that serves it" on docs/llm/llm-boundaries.md.

Offline: the preset registry, and a gateway preset pointed at a local recording
endpoint so the request body can be read.

Most providers are configuration over one wire. A preset's capability flags
change the request quietly: no native response_format → the schema goes into
the system prompt; no parallel_tool_calls → the field is dropped.
"""

import json
import os

import _common  # noqa: F401
from _common import heading
from _recorder import Recorder
from pydantic import BaseModel

from fastaiagent import LLMClient
from fastaiagent.llm import SystemMessage, UserMessage
from fastaiagent.llm.providers import (
    ProviderPreset,
    list_presets,
    list_provider_keys,
    register_provider,
    reserved_keys,
)

heading("what ships")
print("built-ins (code paths):", sorted(reserved_keys()))
print(f"{'preset':<12} {'wire':<14} {'response_format':<16} {'parallel_tool_calls':<20} env var")
for p in list_presets():
    print(f"{p.key:<12} {p.wire:<14} {str(p.cap('response_format')):<16} "
          f"{str(p.cap('parallel_tool_calls')):<20} {p.env_var}")
print("provider keys in total:", len(list_provider_keys()))

heading("a gateway preset in five lines, and what its flags do to the body")
rec = Recorder()
register_provider(ProviderPreset(
    key="corp-gateway", base_url=rec.start(), env_var="CORP_GATEWAY_KEY", default_model="house-7b",
    wire="openai_compat",
    capabilities={"tools": True, "streaming": True, "response_format": False, "parallel_tool_calls": False},
))
os.environ["CORP_GATEWAY_KEY"] = "secret-from-env"


class City(BaseModel):
    name: str
    country: str


TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]
llm = LLMClient(provider="corp-gateway", model="house-7b", max_tokens=64, parallel_tool_calls=True)
r = llm.complete([SystemMessage("Answer briefly."), UserMessage("Which city has the Eiffel Tower?")],
                 tools=TOOLS, output_type=City)
req = rec.last
print("POST", req["path"], "· headers", req["headers"], "(key read from CORP_GATEWAY_KEY)")
print("body keys:", sorted(req["body"]))
print("system message sent:")
print(json.dumps(req["body"]["messages"][0]["content"], ensure_ascii=False)[:400], "…")
print("response_format in body:", "response_format" in req["body"],
      "· parallel_tool_calls in body:", "parallel_tool_calls" in req["body"],
      "· max_tokens key:", "max_tokens" if "max_tokens" in req["body"] else "max_completion_tokens")
print("the canned reply parsed as City:", r.parsed)
rec.stop()
