"""Proof for "6 · Between the call and the numbers" on docs/llm/llm-boundaries.md.

Offline: the pricing table, a local endpoint that answers 429 then 503 then 200,
a one-pixel image, and the SDK's own TestModel for a span.

Cost comes from a local table by longest prefix. Retries are off unless asked
for, and then only on 429 and 5xx with backoff. The span carries a summary of
the request, never the image bytes, and payloads are stripped on export.
"""

import base64
import json
import os
import time

import _common  # noqa: F401
from _common import SpanCollector, heading
from _recorder import Recorder

from fastaiagent import Agent, LLMClient
from fastaiagent._internal.errors import LLMProviderError
from fastaiagent._internal.pricing import compute_cost_usd, is_local_free
from fastaiagent.llm import UserMessage
from fastaiagent.multimodal import Image
from fastaiagent.testing import TestModel
from fastaiagent.trace.redaction import apply_export_policy

heading("cost: a local table, longest prefix wins")
for model in ("gpt-4o-mini-2024-07-18", "gpt-4.1-mini", "claude-sonnet-5-5", "my-private-finetune"):
    print(f"{model:<24} 1M in + 1M out → {compute_cost_usd(model, 1_000_000, 1_000_000)}")
print(f"{'ollama / lmstudio / vllm':<24} is_local_free → {is_local_free('ollama')}, a known zero, not an unknown")

heading("retries: off by default; 429 and 5xx only, with backoff")
rec = Recorder(statuses=[429, 503, 200])
url = rec.start()
t0 = time.monotonic()
r = LLMClient(provider="custom", model="demo", base_url=url, api_key="k", max_retries=2).complete(
    [UserMessage("hi")])
print(f"max_retries=2 : {len(rec.requests)} requests, content={r.content!r}, "
      f"latency_ms={r.latency_ms} (two backoffs: 1 s + 2 s), wall {time.monotonic() - t0:.1f} s")
rec.statuses = [429]
try:
    LLMClient(provider="custom", model="demo", base_url=url, api_key="k").complete([UserMessage("hi")])
except LLMProviderError as e:
    print(f"max_retries=0 : LLMProviderError status={e.status_code}: {str(e)[:60]}…")
rec.stop()

heading("two serializers: the span summary never carries the image")
png = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
msg = UserMessage(["What is this?", Image.from_bytes(png, "image/png")])
print("to_openai_format()  (spans, logs):", json.dumps(msg.to_openai_format()))
wire = msg.to_provider_dict("openai")
image_part = next(p for p in wire["content"] if p.get("type") == "image_url")
print("to_provider_dict()  (the wire)   :", f"image_url = data:image/png;base64,… ({len(image_part['image_url']['url'])} chars)")

heading("the llm span, and what leaves the machine with FASTAIAGENT_TRACE_PAYLOADS=0")
col = SpanCollector()
Agent(name="echo", system_prompt="Echo.", llm=TestModel(response="ok")).run("hello")
attrs = col.llm()[-1]
os.environ["FASTAIAGENT_TRACE_PAYLOADS"] = "0"
exported = apply_export_policy(dict(attrs))
genai = sorted(k for k in attrs if k.startswith("gen_ai."))
print("captured locally:", genai)
print("dropped on export:", sorted(k for k in genai if k not in exported))
