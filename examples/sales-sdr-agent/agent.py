"""
Sales SDR Agent — FastAIAgent SDK Template (Chain DAG with HITL gate)

A Chain-orchestrated sales SDR pipeline:
    enrich → score → branch on qualified → draft → HITL approve → send → log

Usage:
    python agent.py
    python agent.py --prospect carol@megacorp.global
    python agent.py --connect              # export traces to platform

Companion files:
    workflow.py        — Chain definition + system prompts
    tools.py           — pluggable enrichment / CRM / email backends
    eval_suite.py      — golden prospects + scorers
    replay_demo.py     — fork the chain at any node, rerun
    streaming_demo.py  — chain.aexecute with chain trace streaming
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

import fastaiagent as fa

from tools import make_deps
from workflow import build_chain


def _print_pending(pending: dict) -> None:
    """Pretty-print the interrupt context for the human reviewer."""
    print("\n  ┌─ Approval required " + "─" * 38)
    print(f"  │ reason : {pending.get('reason')}")
    info = pending.get("context") or {}
    for k, v in info.items():
        if k == "preview":
            print(f"  │ preview:\n  │   " + str(v).replace("\n", "\n  │   "))
        else:
            print(f"  │ {k:<8}: {v}")
    print("  └" + "─" * 58)


async def run_one(prospect_email: str) -> None:
    chain = build_chain()
    deps = make_deps()
    ctx = fa.RunContext(state=deps)

    execution_id = f"sdr-{uuid.uuid4().hex[:10]}"
    print(f"\nProspect: {prospect_email}")
    print(f"Execution: {execution_id}")
    print("=" * 60)

    initial_state = {"prospect_email": prospect_email}
    result = await chain.aexecute(initial_state, execution_id=execution_id, context=ctx)

    while result.status == "paused":
        _print_pending(result.pending_interrupt or {})
        answer = input("  Approve send? [y/N]: ").strip().lower()
        approved = answer in {"y", "yes"}
        result = await chain.aresume(
            execution_id,
            resume_value=fa.Resume(
                approved=approved,
                metadata={"approver": "cli", "notes": "" if approved else "rejected at CLI"},
            ),
            context=ctx,
        )

    print()
    print("─" * 60)
    print(f"Status:    {result.status}")
    print(f"Final state keys: {sorted(result.final_state.keys())}")
    if result.node_results:
        print(f"Nodes executed:")
        for nid in result.node_results:
            print(f"  • {nid}")
    print(f"\nLast output: {json.dumps(result.output, indent=2, default=str) if isinstance(result.output, dict) else result.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sales SDR Chain Agent")
    parser.add_argument(
        "--prospect",
        default="alice@acme-saas.com",
        help="Prospect email (default: %(default)s — must exist in tools.py mock corpus)",
    )
    parser.add_argument("--connect", action="store_true", help="Connect to FastAIAgent Platform")
    args = parser.parse_args()

    if args.connect:
        api_key = os.getenv("FASTAIAGENT_API_KEY")
        project = os.getenv("FASTAIAGENT_PROJECT", "sales-sdr-agent")
        target = os.getenv("FASTAIAGENT_TARGET", "https://app.fastaiagent.net")
        if api_key:
            fa.connect(api_key=api_key, target=target, project=project)
            print(f"Connected to FastAIAgent Platform (project: {project})")
        else:
            print("FASTAIAGENT_API_KEY not set — running without platform connection")

    asyncio.run(run_one(args.prospect))


if __name__ == "__main__":
    main()
