"""
Personal Assistant — FastAIAgent SDK Template (ComposableMemory showcase).

A long-lived REPL personal assistant that demonstrates every memory-block
type the SDK ships:

  StaticBlock + SummaryBlock + VectorBlock + FactExtractionBlock
                       │
                       ▼
              ComposableMemory
                       │
                       ▼
                  ┌────────┐  + tools: add_note, search_notes,
                  │ Agent  │           list_facts, today
                  └────────┘
                       │
                       ▼
                  REPL session
   (memory.save() on graceful exit; loaded on next startup)

Usage:
    python agent.py
    python agent.py --reset             # wipe saved memory and start fresh
    python agent.py --query "What did we discuss about deployment?"

Companion files:
    memory_setup.py    — wires the four blocks + persistence
    tools.py           — note-taking + introspection tools
    eval_suite.py      — multi-turn session eval
    tests/test_smoke.py — offline tests against the memory pipeline
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import fastaiagent as fa

from memory_setup import build_memory, save_memory
from tools import add_note, list_facts, make_deps, search_notes, today

_HERE = Path(__file__).resolve().parent


# ─── System prompt — loaded from PromptRegistry ──────────────────────────────
# The default below is used to seed the registry on first run. Subsequent
# runs read the current version (you can edit it live in the Local UI's
# Playground at http://127.0.0.1:7842/playground/personal-assistant-prompt
# and the next REPL turn picks up the new version — no restart, no env var).

_DEFAULT_SYSTEM_PROMPT = """You are the user's personal assistant.

You have a layered memory system:
  - A pinned identity block tells you who the user is, their role, and
    their timezone. Use that information when natural; don't ignore it.
  - A rolling summary of older turns is injected automatically.
  - Semantically similar prior exchanges are recalled per turn.
  - Durable facts about the user are extracted and bullet-listed.

You also have tools:
  - ``add_note(text)`` — save a durable note. Use whenever the user asks
    you to remember a specific thing.
  - ``search_notes(query)`` — explicit search over saved notes.
  - ``list_facts()`` — show what the assistant has learned about the user.
  - ``today()`` — get today's ISO date when discussing schedules.

Tone: warm, direct, terse unless the user asks for detail. Don't
preface answers with "Of course!" / "Certainly!". Don't repeat what the
user just said back to them.
"""


_PROMPT_SLUG = "personal-assistant-prompt"


def _load_system_prompt() -> str:
    """Read the system prompt from the local ``PromptRegistry``.

    On first run the default text above is registered as version 1; every
    subsequent run reads the latest version, so a Local-UI Playground edit
    takes effect on the next turn without restarting the process. If the
    registry is unreachable for any reason we fall back to the in-source
    default — keeps the example runnable in fully offline scenarios.
    """
    try:
        registry = fa.PromptRegistry()
        try:
            return registry.get(_PROMPT_SLUG, source="local").template
        except Exception:
            return registry.register(
                name=_PROMPT_SLUG, template=_DEFAULT_SYSTEM_PROMPT
            ).template
    except Exception:
        return _DEFAULT_SYSTEM_PROMPT


def _build_agent(memory) -> fa.Agent:
    return fa.Agent(
        name="personal-assistant",
        # Callable system_prompt → the Agent re-resolves it on every turn,
        # so live edits in the UI Playground propagate without restart.
        system_prompt=lambda _ctx=None: _load_system_prompt(),
        llm=fa.LLMClient(provider="openai", model=os.getenv("LLM_MODEL", "gpt-4o")),
        tools=[add_note, search_notes, list_facts, today],
        memory=memory,
    )


def _resolve_memory_dir() -> Path:
    raw = os.getenv("MEMORY_DIR", ".fastaiagent/memory")
    p = Path(raw)
    return p if p.is_absolute() else (_HERE / p)


async def _drive_query(query: str) -> None:
    """Single-shot path. Loads memory, answers once, persists, exits."""
    memory_dir = _resolve_memory_dir()
    memory = build_memory(memory_dir=memory_dir)
    deps = make_deps(memory=memory)
    ctx = fa.RunContext(state=deps)

    agent = _build_agent(memory)
    result = await agent.arun(query, context=ctx)
    print(f"\nAssistant: {result.output}\n")
    print(f"  {result.tokens_used} tokens | ${result.cost:.4f} | {result.latency_ms} ms")
    save_memory(memory, memory_dir)


async def _drive_repl() -> None:
    memory_dir = _resolve_memory_dir()
    memory = build_memory(memory_dir=memory_dir)
    deps = make_deps(memory=memory)
    ctx = fa.RunContext(state=deps)
    agent = _build_agent(memory)

    print("\n" + "=" * 56)
    print("  Personal Assistant")
    print(f"  Memory dir: {memory_dir}")
    print("  Type 'quit' to exit (memory is saved automatically).")
    print("=" * 56 + "\n")

    try:
        while True:
            try:
                user = input("You: ").strip()
            except EOFError:
                break
            if not user:
                continue
            if user.lower() in ("quit", "exit"):
                break
            result = await agent.arun(user, context=ctx)
            print(f"\nAssistant: {result.output}\n")
    finally:
        save_memory(memory, memory_dir)
        print(f"  ✓ memory saved to {memory_dir}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Personal Assistant")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe the on-disk memory and start fresh")
    parser.add_argument("--query", type=str, help="Single-shot query, then exit")
    parser.add_argument("--connect", action="store_true",
                        help="Connect to FastAIAgent Platform")
    args = parser.parse_args()

    if args.reset:
        memory_dir = _resolve_memory_dir()
        if memory_dir.exists():
            shutil.rmtree(memory_dir)
            print(f"  ✓ wiped {memory_dir}")
        else:
            print("  (memory dir didn't exist; nothing to wipe)")

    if args.connect:
        api_key = os.getenv("FASTAIAGENT_API_KEY")
        project = os.getenv("FASTAIAGENT_PROJECT", "personal-assistant")
        target = os.getenv("FASTAIAGENT_TARGET", "https://app.fastaiagent.net")
        if api_key:
            fa.connect(api_key=api_key, target=target, project=project)
            print(f"Connected to FastAIAgent Platform (project: {project})")

    if args.query:
        asyncio.run(_drive_query(args.query))
    else:
        asyncio.run(_drive_repl())


if __name__ == "__main__":
    main()
