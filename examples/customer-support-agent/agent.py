"""
Customer Support Agent — FastAIAgent SDK Template (v1.6.0)

A production-shaped support agent demonstrating the v1.6.0 capability surface:

  * Tools + LocalKB hybrid search (FAISS + BM25)
  * RunContext[Deps] dependency injection
  * Built-in guardrails (PII output filter, toxicity input filter)
  * AgentMemory across REPL turns                       (v0.x)
  * Middleware: ToolBudget + TrimLongMessages           (v0.x / hardened in v1.5.0)
  * PromptRegistry-backed system prompt (editable in Local UI Playground, v1.3.0)
  * SQLiteCheckpointer + interrupt()/aresume()          (v1.0)
  * Optional fa.connect() to export traces              (since v0.1)

Usage:
    python agent.py
    python agent.py --connect          # with platform traces
    python agent.py --query "I can't log in to my account"

Companion demos in this folder:
    python streaming_demo.py     # agent.astream() token-by-token
    python replay_demo.py        # fork-and-rerun with fa.Replay
    python multimodal_demo.py    # screenshot + text via fa.Image
    python eval_suite.py         # LLM-judge + RAG scorers
"""

import argparse
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()  # must run before SDK/local imports so env vars are available

import fastaiagent as fa
from fastaiagent.agent.memory import AgentMemory
from fastaiagent.agent.middleware import ToolBudget, TrimLongMessages

from context import Deps, create_deps
from guardrails import pii_filter, toxicity_check
from tools import check_order_status, create_ticket, lookup_account, search_kb

# ─── System Prompt (PromptRegistry-backed) ───────────────────────────────────
# Stored in `.fastaiagent/local.db` so the prompt is editable from the Local
# UI Playground at http://localhost:8765/playground (v1.3.0). On first run we
# register the default; subsequent runs read the latest version.

_DEFAULT_SYSTEM_PROMPT = """You are a friendly, professional customer support agent for TechCorp.

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


def _load_system_prompt() -> str:
    registry = fa.PromptRegistry()
    try:
        return registry.get("support-system-prompt", source="local").template
    except Exception:
        return registry.register(
            name="support-system-prompt",
            template=_DEFAULT_SYSTEM_PROMPT,
        ).template


SYSTEM_PROMPT = _load_system_prompt()

# ─── Agent Definition ────────────────────────────────────────────────────────

# AgentMemory persists across every arun() on this Agent instance — so the REPL
# now remembers prior turns within a session. Cleared at process exit.
_memory = AgentMemory(max_messages=20)

# SQLiteCheckpointer enables interrupt()/aresume(). The default DB path lives
# under `.fastaiagent/local.db` alongside traces.
_checkpointer = fa.SQLiteCheckpointer()

agent = fa.Agent(
    name="customer-support",
    llm=fa.LLMClient(provider="openai", model=os.getenv("LLM_MODEL", "gpt-4o")),
    system_prompt=SYSTEM_PROMPT,
    tools=[search_kb, create_ticket, lookup_account, check_order_status],
    guardrails=[pii_filter, toxicity_check],
    memory=_memory,
    middleware=[
        ToolBudget(max_calls=10, message="Hit the tool-call budget — escalating to a human."),
        TrimLongMessages(keep_last=20),
    ],
    checkpointer=_checkpointer,
)


# ─── HITL helper ─────────────────────────────────────────────────────────────


async def _drive_until_complete(query: str, ctx: fa.RunContext[Deps]) -> fa.AgentResult:
    """Run the agent, prompting the human on any pending interrupt() and
    looping aresume() until the run reaches a terminal status."""
    result = await agent.arun(query, context=ctx)
    while result.status == "paused":
        info = result.pending_interrupt or {}
        print("\n  ┌─ Approval required ──────────────────────────────")
        print(f"  │ reason : {info.get('reason')}")
        for k, v in (info.get("context") or {}).items():
            print(f"  │ {k:<8}: {v}")
        print("  └──────────────────────────────────────────────────")
        answer = input("  Approve? [y/N]: ").strip().lower()
        approved = answer in {"y", "yes"}
        result = await agent.aresume(
            result.execution_id,
            resume_value=fa.Resume(approved=approved, metadata={"approver": "cli"}),
            context=ctx,
        )
    return result


# ─── Runners ─────────────────────────────────────────────────────────────────


async def run_interactive() -> None:
    """REPL with multi-turn memory + HITL approvals."""
    deps = await create_deps()
    ctx = fa.RunContext(state=deps)

    print("\n" + "=" * 60)
    print("  Customer Support — TechCorp")
    print("  Memory persists across turns. Type 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            break

        result = await _drive_until_complete(query, ctx)
        print(f"\nAgent: {result.output}\n")
        print(f"  {result.tokens_used} tokens | ${result.cost:.4f} | {result.latency_ms}ms\n")


async def run_single(query: str) -> None:
    deps = await create_deps()
    ctx = fa.RunContext(state=deps)
    result = await _drive_until_complete(query, ctx)
    print(f"\nAgent: {result.output}\n")
    print(f"  {result.tokens_used} tokens | ${result.cost:.4f} | {result.latency_ms}ms")


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Customer Support Agent")
    parser.add_argument("--connect", action="store_true", help="Connect to FastAIAgent Platform")
    parser.add_argument("--query", type=str, help="Single query (non-interactive)")
    args = parser.parse_args()

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
