"""
Streaming Demo — supervisor.astream() with handoff visibility.

Run: python streaming_demo.py --topic "Constitutional AI"

The Supervisor's ``astream()`` yields ``TextDelta`` events for the
supervisor's *own* synthesis tokens AND ``ToolCallStart`` / ``ToolCallEnd``
events every time the supervisor delegates to a worker. Worker execution
itself is not streamed — but you can see the handoff boundaries in real time.

That's what makes this useful for ops: a researcher worker that takes 4
seconds shows up as a single ``ToolCallStart(delegate_to_researcher)`` →
``ToolCallEnd(...)`` pair in the stream, so the operator knows the
supervisor is alive even while the worker is grinding.
"""

from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

load_dotenv()

import fastaiagent as fa

from tools import make_deps
from topology import build_supervisor


async def stream_topic(topic: str) -> None:
    deps = make_deps()
    ctx = fa.RunContext(state=deps)

    supervisor = build_supervisor()

    print(f"\nTopic: {topic}")
    print("=" * 60)

    handoffs: list[tuple[str, str]] = []  # (event_kind, tool_name)

    async for event in supervisor.astream(topic, context=ctx):
        if isinstance(event, fa.TextDelta):
            print(event.text, end="", flush=True)
        elif type(event).__name__ == "ToolCallStart":
            tool_name = getattr(event, "tool_name", "?")
            handoffs.append(("start", tool_name))
            print(f"\n  ▷ handoff: {tool_name}", flush=True)
        elif type(event).__name__ == "ToolCallEnd":
            tool_name = getattr(event, "tool_name", "?")
            handoffs.append(("end", tool_name))
            print(f"\n  ◁ done:    {tool_name}", flush=True)

    print()
    print("─" * 60)
    print(f"  total handoffs: {sum(1 for kind, _ in handoffs if kind == 'start')}")
    print(f"  retrieved sources: {len(deps.trail)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topic",
        default="Transformer architecture",
        help="Research topic (default: %(default)r)",
    )
    args = parser.parse_args()
    asyncio.run(stream_topic(args.topic))


if __name__ == "__main__":
    main()
