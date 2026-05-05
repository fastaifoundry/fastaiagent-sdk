"""
Multimodal Demo — accepting screenshots alongside text.

Run: python multimodal_demo.py path/to/screenshot.png "I'm seeing this error"
     python multimodal_demo.py --url https://example.com/error.png "What does this mean?"

Customers often send screenshots of error dialogs. ``fa.Image`` + ``fa.PDF``
(v1.1.0) are first-class inputs to ``agent.arun(...)`` — the LLMClient handles
provider-specific wire formatting (OpenAI vision parts, Anthropic blocks, etc.).

We construct a checkpointer-free Agent here because the v1.6.0
``SQLiteCheckpointer`` does not yet serialize raw image bytes inside
turn-boundary checkpoints. Multimodal + HITL together is a future-work item;
for now, demo each capability independently.
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

import fastaiagent as fa

from agent import SYSTEM_PROMPT
from context import Deps, create_deps
from guardrails import pii_filter, toxicity_check
from tools import check_order_status, lookup_account, search_kb

# Note: no `create_ticket` tool here — interrupt() requires a checkpointer,
# which we omit for multimodal compatibility. If you need ticket creation
# in a multimodal flow, pass the screenshot to a human agent instead.

multimodal_agent = fa.Agent(
    name="customer-support-multimodal",
    llm=fa.LLMClient(provider="openai", model=os.getenv("LLM_MODEL", "gpt-4o")),
    system_prompt=SYSTEM_PROMPT,
    tools=[search_kb, lookup_account, check_order_status],
    guardrails=[pii_filter, toxicity_check],
)


async def run_with_image(query: str, image: fa.Image) -> None:
    deps = await create_deps()
    ctx: fa.RunContext[Deps] = fa.RunContext(state=deps)

    result = await multimodal_agent.arun([query, image], context=ctx)
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
