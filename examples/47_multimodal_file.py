"""Example 47: Multimodal — the generic ``File`` input (any bytes, natively).

Hand the agent raw bytes or a ``File`` and the SDK forwards it to the model
using each provider's *native* file mechanism — no local rendering, no
stringifying. Bare ``bytes`` and ``Path`` inputs are auto-detected (mime
sniffed) and wrapped in a ``File`` for you.

This demo sends the same PDF three ways:

* OpenAI gpt-4o     — auto-detect from bare ``bytes`` → native ``file`` part
* Anthropic Sonnet  — ``File.from_bytes`` → ``document`` block
* Gemini 2.5-flash  — ``File`` → ``inlineData``

Usage::

    zsh -lc 'export OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... GEMINI_API_KEY=...'
    zsh -lc 'python examples/47_multimodal_file.py'

The fixture PDF is regenerated on demand via
``tests/fixtures/multimodal/_make_fixtures.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastaiagent import Agent, File, LLMClient

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "multimodal"
QUESTION = "What is the term length of this agreement? Answer in a few words."


def _pdf_bytes() -> bytes:
    pdf_path = FIXTURES / "contract.pdf"
    if not pdf_path.exists():
        from tests.fixtures.multimodal._make_fixtures import main as make

        make()
    return pdf_path.read_bytes()


def ask(label: str, agent: Agent, content: list[object]) -> None:
    result = agent.run(content)
    print(f"--- {label} ---")
    print(f"Output: {result.output}")
    print(f"Tokens: {result.tokens_used}\n")


if __name__ == "__main__":
    pdf = _pdf_bytes()

    if os.environ.get("OPENAI_API_KEY"):
        ask(
            "openai/gpt-4o — bare bytes (auto-detect → native file part)",
            Agent(name="f-openai", system_prompt="Answer from the document.",
                  llm=LLMClient(provider="openai", model="gpt-4o")),
            [QUESTION, pdf],  # <-- raw bytes; the SDK sniffs + wraps in File
        )
    else:
        print("Skipping OpenAI: OPENAI_API_KEY not set")

    if os.environ.get("ANTHROPIC_API_KEY"):
        ask(
            "anthropic/claude-sonnet-4 — File.from_bytes (→ document block)",
            Agent(name="f-anthropic", system_prompt="Answer from the document.",
                  llm=LLMClient(provider="anthropic", model="claude-sonnet-4-6")),
            [QUESTION, File.from_bytes(pdf, filename="contract.pdf")],
        )
    else:
        print("Skipping Anthropic: ANTHROPIC_API_KEY not set")

    if os.environ.get("GEMINI_API_KEY"):
        ask(
            "gemini/2.5-flash — File (→ inlineData)",
            Agent(name="f-gemini", system_prompt="Answer from the document.",
                  llm=LLMClient(provider="gemini", model="gemini-2.5-flash")),
            [QUESTION, File.from_bytes(pdf)],
        )
    else:
        print("Skipping Gemini: GEMINI_API_KEY not set")
