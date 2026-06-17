"""End-to-end multimodal gate — Anthropic Claude.

Real Anthropic API. No mocking. Hits the live endpoint using
``ANTHROPIC_API_KEY``. Exercises:

* Image input → Claude vision response
* PDF native-mode (Anthropic's ``document`` block) → contract Q&A

Run with::

    zsh -lc 'pytest tests/e2e/test_gate_multimodal_anthropic.py -m e2e -v'
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from fastaiagent import PDF, Agent, Image, LLMClient
from tests.e2e.conftest import require_anthropic, require_env

pytestmark = pytest.mark.e2e

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "multimodal"

# Pinned to a current Sonnet model that supports vision and native PDF.
ANTHROPIC_MODEL = "claude-sonnet-4-6"


def _ensure_fixtures() -> None:
    if not (FIXTURES / "cat.jpg").exists():
        from tests.fixtures.multimodal._make_fixtures import main as make

        make()


def _claude_agent(name: str = "vision-test") -> Agent:
    return Agent(
        name=name,
        system_prompt=(
            "You are a vision assistant. Be concise. "
            "If you see text in an image, quote it exactly."
        ),
        llm=LLMClient(provider="anthropic", model=ANTHROPIC_MODEL),
    )


def _reports_cat(output: str) -> bool:
    """True if Claude's vision answer identifies the letters C-A-T.

    Robust to phrasing: Claude may answer "CAT", "C A T", or spell it out with
    connectors/markdown ("C, A, and T", "**C**, **A**, **T**"). We accept either
    the contiguous word or the three letters surfaced as standalone tokens, so
    the test checks comprehension rather than exact wording.
    """
    upper = output.upper()
    contiguous = "".join(c for c in upper if c.isalpha())
    spelled = "".join(re.findall(r"\b[A-Z]\b", upper))  # "C, A, and T" -> "CAT"
    return "CAT" in contiguous or "CAT" in spelled


class TestAnthropicVisionGate:
    """Spec test #1, #5, native-PDF — real Claude vision."""

    def test_image_input_round_trip(self) -> None:
        require_env()
        require_anthropic()
        _ensure_fixtures()
        agent = _claude_agent("describe-image")
        result = agent.run(
            ["What letters appear in this image?", Image.from_file(FIXTURES / "cat.jpg")]
        )
        assert result.output, "expected non-empty output from Claude"
        # Claude may return "CAT", "C A T", or spell it out ("C, A, and T") —
        # all indicate it read the image correctly.
        assert _reports_cat(result.output), (
            f"Claude vision must report CAT (any phrasing); got: {result.output[:200]}"
        )

    def test_native_pdf_block_contract_q_and_a(self) -> None:
        """Anthropic accepts native PDF — verify our auto-mode emits a document block
        and Claude answers from the document."""
        require_env()
        require_anthropic()
        _ensure_fixtures()
        agent = _claude_agent("native-pdf")
        result = agent.run(
            [
                "What is the term length of this agreement?",
                PDF.from_file(FIXTURES / "contract.pdf"),
            ]
        )
        assert "two years" in result.output.lower() or "2 years" in result.output.lower()

    def test_mixed_image_and_pdf_in_one_call(self) -> None:
        """Spec test #5 — text + image + native PDF in one Anthropic call."""
        require_env()
        require_anthropic()
        _ensure_fixtures()
        agent = _claude_agent("mixed-input")
        result = agent.run(
            [
                "I will give you an image and a contract. "
                "First report the letters in the image, then the contract's term length. "
                "Format: 'IMAGE: <letters>; CONTRACT: <duration>'.",
                Image.from_file(FIXTURES / "cat.jpg"),
                PDF.from_file(FIXTURES / "contract.pdf"),
            ]
        )
        assert _reports_cat(result.output)
        assert "two years" in result.output.lower() or "2 years" in result.output.lower()
