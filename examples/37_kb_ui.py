"""Example 37 — Local UI knowledge-base browser.

Builds a small ``LocalKB`` from three hand-written policy docs, runs an agent
that uses it as a tool so the retrieval shows up in the Lineage tab, then
points you at the UI.

Prereqs:
    pip install 'fastaiagent[ui,openai,kb]'
    export OPENAI_API_KEY=...

Run:
    python examples/37_kb_ui.py

Then:
    fastaiagent ui
    open http://127.0.0.1:7842/kb
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("FASTAIAGENT_UI_ENABLED", "true")

from fastaiagent import Agent, LLMClient  # noqa: E402
from fastaiagent.kb import LocalKB  # noqa: E402

_DOCS = {
    "refund-policy.md": (
        "# Refund policy\n\n"
        "Refunds are processed within 7 business days after we receive the "
        "return. Customers must include the original packing slip. Items "
        "marked final-sale are non-refundable.\n"
    ),
    "shipping-policy.md": (
        "# Shipping policy\n\n"
        "Standard shipping is free on orders over $50. Expedited shipping is "
        "available at checkout for an extra charge.\n"
    ),
    "return-window.md": (
        "# Return window\n\n"
        "You have 30 days from the delivery date to initiate a return. "
        "Start a return from the order history page in your account.\n"
    ),
}


def _seed_kb() -> LocalKB:
    source_dir = Path(".fastaiagent-source-docs")
    source_dir.mkdir(exist_ok=True)
    for name, content in _DOCS.items():
        (source_dir / name).write_text(content)

    kb = LocalKB(name="support-kb", chunk_size=240, chunk_overlap=30)
    for name in _DOCS:
        kb.add(str(source_dir / name))
    print(f"► seeded LocalKB 'support-kb' with {len(_DOCS)} documents")
    return kb


def _run_agent_with_kb(kb: LocalKB) -> None:
    llm = LLMClient(provider="openai", model="gpt-4o-mini")
    agent = Agent(
        name="support-bot",
        system_prompt=(
            "You are a terse customer-support assistant. Always search the "
            "knowledge base before answering policy questions."
        ),
        llm=llm,
        tools=[kb.as_tool()],
    )
    for query in [
        "When do refunds get processed?",
        "How long do I have to start a return?",
        "Is shipping free on a $45 order?",
    ]:
        print(f"\n► asking: {query}")
        result = agent.run(query)
        print(f"  → {result.output[:200]}")


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required to run this example.")

    kb = _seed_kb()
    _run_agent_with_kb(kb)

    print()
    print("Done. Now open the UI:")
    print("    fastaiagent ui")
    print()
    print("What you'll see:")
    print("  • /kb              — cards for every LocalKB collection found")
    print("  • /kb/support-kb   — Documents, Search playground, Lineage tabs")
    print("  • /traces          — the 3 agent runs that hit the KB")
    print()
    print("On /kb/support-kb try searching 'refund policy' — the UI calls the")
    print("same kb.search() you used from code, and the Lineage tab shows the")
    print("agent runs that just retrieved from it.")


if __name__ == "__main__":
    main()
