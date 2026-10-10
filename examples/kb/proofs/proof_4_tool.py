"""Proof for "4 · Between the knowledge base and the model" on docs/knowledge-base/kb-boundaries.md.

Live: needs OPENAI_API_KEY (gpt-4.1-mini) and the local FastEmbed model.

Retrieval is a tool call. The model writes the query, the tool returns scored
chunks as text, and the trace keeps a retrieval span with the query and the ids
that came back. Local capture is full fidelity; the export policy strips the
payload-bearing keys when FASTAIAGENT_TRACE_PAYLOADS=0.
"""

import json
import os
import sys

import _common
from _common import SpanCollector, embedder, heading, retrieval_spans

from fastaiagent import Agent, LLMClient
from fastaiagent.kb import LocalKB

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY is not set — this proof calls a real model.")

col = SpanCollector()
kb = LocalKB(name="support-kb", path=str(_common.KB_DIR), embedder=embedder(), persist=False)
for d in (
    "Items can be sent back within 30 days of purchase for a full refund.",
    "Error code ERR-4012 means the payment gateway timed out. Retry after 30 seconds, "
    "then contact support with the transaction id.",
    "Support hours are Monday to Friday, 9am to 5pm EST.",
):
    kb.add(d)

tool = kb.as_tool()
heading("the tool the model sees")
print(f"name={tool.name!r} origin={tool.origin!r} description={tool.description!r}")

agent = Agent(
    name="support-bot",
    system_prompt="Answer from the knowledge base. Search before you answer, and quote what you found.",
    llm=LLMClient(provider="openai", model=os.environ.get("PROOF_MODEL", "gpt-4.1-mini"), temperature=0),
    tools=[tool],
)
result = agent.run("My checkout shows ERR-4012. What do I do?")

heading("what happened, from the spans")
for name, attrs in col.spans:
    if name.startswith("tool."):
        for k, v in attrs.items():
            if k in ("tool.name", "tool.args", "tool.result", "tool.output", "tool.origin"):
                print(f"{name}: {k}={str(v)[:120]!r}")
for attrs in retrieval_spans(col):
    keep = {k.removeprefix("retrieval."): v for k, v in attrs.items() if k.startswith("retrieval.")}
    print("retrieval span:", json.dumps(keep)[:300])
print("answer:", result.output[:160].replace("\n", " "))

heading("the export policy, with FASTAIAGENT_TRACE_PAYLOADS=0")
from fastaiagent.trace.redaction import apply_export_policy  # noqa: E402

os.environ["FASTAIAGENT_TRACE_PAYLOADS"] = "0"
attrs = retrieval_spans(col)[-1]
exported = apply_export_policy(dict(attrs))
print("kept   :", sorted(k for k in exported if k.startswith("retrieval.")))
print("dropped:", sorted(k for k in attrs if k.startswith("retrieval.") and k not in exported))
