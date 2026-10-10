"""Proof for "1 · Between calls: the tree is discovered" on docs/tracing/trace-boundaries.md.

Offline: the SDK's own FunctionModel, no API key, a throwaway local.db.

Two agents run at the same time in one process. Every span is stored as a flat
row with its parent_span_id; the tree is rebuilt from those links. A span opened
inside a tool body attaches under the tool span with no plumbing. Neither run's
spans land in the other's tree.
"""

import asyncio
import os
import tempfile

os.environ["FASTAIAGENT_LOCAL_DB"] = os.path.join(tempfile.mkdtemp(), "local.db")

import fastaiagent as fa  # noqa: E402
from fastaiagent.testing import FunctionModel  # noqa: E402
from fastaiagent.trace import TraceStore, trace_context  # noqa: E402


def make_agent(name: str, carrier: str) -> fa.Agent:
    @fa.tool(name=f"lookup_{name}")
    def lookup(order_id: str) -> str:
        """Look an order up."""
        with trace_context("db.query", table="orders"):  # opened inside the tool body
            return f"{order_id} shipped via {carrier}"

    def responder(messages):
        if messages[-1].role.value != "tool":  # turn 1: ask for the tool
            return "", [{"name": f"lookup_{name}", "arguments": {"order_id": "1042"}}]
        return f"Order {messages[-1].content}."  # turn 2: answer from the tool result

    return fa.Agent(name=name, llm=FunctionModel(responder), tools=[lookup])


async def main() -> None:
    alpha, beta = make_agent("alpha", "DHL"), make_agent("beta", "UPS")
    ra, rb = await asyncio.gather(
        alpha.arun("Where is order 1042?"), beta.arun("Where is order 1042?")
    )
    store = TraceStore()
    for label, result in (("alpha", ra), ("beta", rb)):
        trace = store.get_trace(result.trace_id)
        roots = [s for s in trace.spans if not s.parent_span_id]
        print(
            f"trace of {label}: {len(trace.spans)} flat rows, "
            f"{len(roots)} parentless root ({roots[0].name})"
        )
        children: dict = {}
        for s in trace.spans:
            children.setdefault(s.parent_span_id or None, []).append(s)

        def walk(parent_id, depth):
            for s in sorted(children.get(parent_id, []), key=lambda s: s.start_time):
                kind = s.attributes.get("fastaiagent.runner.type", "—")
                print(f"  {'    ' * depth}{s.name:<28} runner.type={kind}")
                walk(s.span_id, depth + 1)

        walk(None, 0)
        print(f"  answer: {result.output!r}")
        other = "beta" if label == "alpha" else "alpha"
        assert not any(other in s.name for s in trace.spans)
        print(f"✓ nothing of {other}'s run is in {label}'s trace")


asyncio.run(main())
