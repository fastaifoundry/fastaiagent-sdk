"""
Multimodal Demo — accepting screenshots alongside text.

Run: python multimodal_demo.py path/to/screenshot.png "I'm seeing this error"
     python multimodal_demo.py --url https://example.com/error.png "What does this mean?"

Customers often send screenshots of error dialogs. ``fa.Image`` + ``fa.PDF``
(v1.1.0) are first-class inputs to ``agent.arun(...)`` — the LLMClient handles
provider-specific wire formatting (OpenAI vision parts, Anthropic blocks, etc.).

This demo uses the same fully-wired ``agent`` from ``agent.py`` — multi-turn
memory, middleware, guardrails, HITL, ``SQLiteCheckpointer``, the lot. Pass a
list of parts (text + Image + PDF) as the input; resumable durability for
multimodal turns ships in v1.6.1+.
"""

import argparse
import asyncio

from dotenv import load_dotenv

load_dotenv()

import fastaiagent as fa

from agent import agent
from context import create_deps


async def run_with_image(query: str, image: fa.Image) -> None:
    deps = await create_deps()
    ctx = fa.RunContext(state=deps)

    result = await agent.arun([query, image], context=ctx)
    print(f"\nAgent: {result.output}\n")
    print(f"  {result.tokens_used} tokens | ${result.cost:.4f} | {result.latency_ms}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a screenshot + question to the support agent")
    parser.add_argument("path", nargs="?", help="Local path to an image file")
    parser.add_argument("query", nargs="?", default="What does this error mean?", help="Question to ask")
    parser.add_argument("--url", help="HTTPS URL of an image instead of a local path")
    args = parser.parse_args()

    if args.url:
        image = fa.Image.from_url(args.url)
    elif args.path:
        image = fa.Image.from_file(args.path)
    else:
        parser.error("Provide either a local image path or --url <https://...>")

    asyncio.run(run_with_image(args.query, image))


if __name__ == "__main__":
    main()
