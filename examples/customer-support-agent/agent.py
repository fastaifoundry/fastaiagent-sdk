"""
Customer Support Agent — FastAIAgent SDK Template

A production-ready support agent with KB search, ticket creation,
guardrails, and platform integration.

Usage:
    python agent.py
    python agent.py --connect          # with platform traces
    python agent.py --query "I can't log in to my account"
"""

import asyncio
import argparse
import os
from dotenv import load_dotenv

load_dotenv()  # must run before SDK/local imports so env vars are available

import fastaiagent as fa

from context import Deps, create_deps
from tools import search_kb, create_ticket, lookup_account, check_order_status
from guardrails import pii_filter, toxicity_check

# ─── System Prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a friendly, professional customer support agent for TechCorp.

Your responsibilities:
1. Answer product questions using the knowledge base (always search first)
2. Help with account issues by looking up the customer's account
3. Create support tickets for issues you cannot resolve directly
4. Check order status when asked about deliveries

Guidelines:
- Always search the knowledge base before answering product questions
- Be concise but thorough — aim for 2-3 sentences unless more detail is needed
- If you don't know something, say so and offer to create a ticket
- Never share sensitive account details like passwords or full payment info
- For billing disputes or refunds, always create a ticket for the billing team
"""

# ─── Agent Definition ────────────────────────────────────────────────────────

agent = fa.Agent(
    name="customer-support",
    llm=fa.LLMClient(provider="openai", model=os.getenv("LLM_MODEL", "gpt-4o")),
    system_prompt=SYSTEM_PROMPT,
    tools=[search_kb, create_ticket, lookup_account, check_order_status],
    guardrails=[pii_filter, toxicity_check],
)

# ─── Interactive Runner ──────────────────────────────────────────────────────

async def run_interactive():
    """Run the agent in interactive chat mode."""
    deps = await create_deps()
    ctx = fa.RunContext(state=deps)

    print("\n" + "=" * 60)
    print("  Customer Support — TechCorp")
    print("  Type 'quit' to exit")
    print("=" * 60 + "\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            break

        result = await agent.arun(query, context=ctx)
        print(f"\nAgent: {result.output}\n")
        print(f"  {result.tokens_used} tokens | ${result.cost:.4f} | {result.latency_ms}ms\n")


async def run_single(query: str):
    """Run a single query and print the result."""
    deps = await create_deps()
    ctx = fa.RunContext(state=deps)
    result = await agent.arun(query, context=ctx)
    print(f"\nAgent: {result.output}\n")
    print(f"  {result.tokens_used} tokens | ${result.cost:.4f} | {result.latency_ms}ms")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Customer Support Agent")
    parser.add_argument("--connect", action="store_true", help="Connect to FastAIAgent Platform")
    parser.add_argument("--query", type=str, help="Single query (non-interactive)")
    args = parser.parse_args()

    # Connect to platform if requested
    if args.connect:
        api_key = os.getenv("FASTAIAGENT_API_KEY")
        project = os.getenv("FASTAIAGENT_PROJECT", "support-bot")
        target = os.getenv("FASTAIAGENT_TARGET", "https://app.fastaiagent.net")
        if api_key:
            fa.connect(api_key=api_key, target=target, project=project)
            print(f"Connected to FastAIAgent Platform (project: {project})")
        else:
            print("FASTAIAGENT_API_KEY not set — running without platform connection")

    if args.query:
        asyncio.run(run_single(args.query))
    else:
        asyncio.run(run_interactive())


if __name__ == "__main__":
    main()
