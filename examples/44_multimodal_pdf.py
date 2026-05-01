"""Example 44: Multimodal — PDF in text mode vs vision mode.

Demonstrates the three ``pdf_mode`` settings against a 2-page contract:

* ``text``    — pymupdf extracts text, sends as plain text. Cheapest, no
                vision LLM required (gpt-3.5-turbo / claude-2.x work).
* ``vision``  — pages rendered to PNG, sent as image blocks to a vision
                LLM. Preserves layout (tables, signatures).
* ``native``  — Anthropic-only ``document`` block. Lowest cost while
                preserving layout — Claude reads the PDF directly.

Token counts are printed so the cost difference is concrete.

Usage::

    zsh -lc 'export OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-...'
    zsh -lc 'python examples/44_multimodal_pdf.py'
"""

from __future__ import annotations

import os
from pathlib import Path

from fastaiagent import PDF, Agent, LLMClient

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "multimodal"


def _ensure_fixture() -> Path:
    pdf_path = FIXTURES / "contract.pdf"
    if not pdf_path.exists():
        from tests.fixtures.multimodal._make_fixtures import main as make

        make()
    return pdf_path


def ask(label: str, agent: Agent, pdf: PDF) -> None:
    result = agent.run(
        ["What is the term length of this agreement?", pdf]
    )
    print(f"--- {label} ---")
    print(f"Output: {result.output}")
    print(f"Tokens used: {result.tokens_used}")
    print(f"Latency:     {result.latency_ms} ms")
    print()


if __name__ == "__main__":
    pdf_path = _ensure_fixture()
    pdf = PDF.from_file(pdf_path)

    if os.environ.get("OPENAI_API_KEY"):
        ask(
            "openai/gpt-3.5-turbo + pdf_mode='text'",
            Agent(
                name="pdf-text",
                system_prompt="Answer concisely from the document.",
                llm=LLMClient(provider="openai", model="gpt-3.5-turbo", pdf_mode="text"),
            ),
            pdf,
        )
        ask(
            "openai/gpt-4o + pdf_mode='vision'",
            Agent(
                name="pdf-vision",
                system_prompt="Answer concisely from the document.",
                llm=LLMClient(provider="openai", model="gpt-4o", pdf_mode="vision"),
            ),
            pdf,
        )
    else:
        print("Skipping OpenAI: OPENAI_API_KEY not set")

    if os.environ.get("ANTHROPIC_API_KEY"):
        ask(
            "anthropic/claude-sonnet-4 + pdf_mode='native' (auto)",
            Agent(
                name="pdf-native",
                system_prompt="Answer concisely from the document.",
                llm=LLMClient(provider="anthropic", model="claude-sonnet-4-20250514"),
            ),
            pdf,
        )
    else:
        print("Skipping Anthropic: ANTHROPIC_API_KEY not set")
