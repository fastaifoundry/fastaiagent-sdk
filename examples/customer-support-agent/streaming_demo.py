"""
Streaming Demo — token-by-token output via ``agent.astream()``.

Run: python streaming_demo.py
     python streaming_demo.py --query "How do I upgrade my plan?"

``agent.astream()`` (v1.5.0) yields ``StreamEvent`` objects as the LLM produces
them. Middleware, guardrails, and tool calls all run identically to ``arun()``.

Note on HITL: ``interrupt()`` raised from inside a tool during streaming
records the suspension and propagates as an exception (the streaming surface
doesn't return a paused ``AgentResult``). For flows that may suspend (e.g.
``create_ticket`` with priority="urgent"), use ``agent.arun()`` — see
``agent.py``'s ``_drive_until_complete`` for the resume-loop pattern.
"""

import argparse
import asyncio

from dotenv import load_dotenv

load_dotenv()

import fastaiagent as fa

from agent import agent
from context import create_deps


async def stream_one(query: str) -> None:
    deps = await create_deps()
    ctx = fa.RunContext(state=deps)

    print(f"\nYou: {query}")
    print("Agent: ", end="", flush=True)

    text_chars = 0
    tool_invocations: list[str] = []
    async for event in agent.astream(query, context=ctx):
        if isinstance(event, fa.TextDelta):
            print(event.text, end="", flush=True)
            text_chars += len(event.text)
        elif type(event).__name__ == "ToolCallStart":
            tool_invocations.append(getattr(event, "tool_name", "?"))

    print()  # newline after stream completes
    if tool_invocations:
        print(f"  tools used: {', '.join(tool_invocations)}")
    print(f"  streamed {text_chars} chars")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="What is your refund policy?",
        help="Question to stream",
    )
    args = parser.parse_args()
    asyncio.run(stream_one(args.query))


if __name__ == "__main__":
    main()
