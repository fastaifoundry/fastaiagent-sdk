"""End-to-end multimodal gate — OpenAI gpt-4o.

Real Anthropic SDK / no — real OpenAI. No mocking. Hits the live API using
``OPENAI_API_KEY`` from the user's environment. The fixture image renders
the literal text "CAT" so the LLM response is reliably checkable without
needing an actual photo.

Run with::

    zsh -lc 'pytest tests/e2e/test_gate_multimodal_openai.py -m e2e -v'
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastaiagent import PDF, Agent, FunctionTool, Image, LLMClient
from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "multimodal"


def _ensure_fixtures() -> None:
    """Auto-generate the fixture binaries if a fresh checkout is missing them."""
    if not (FIXTURES / "cat.jpg").exists():
        from tests.fixtures.multimodal._make_fixtures import main as make

        make()


def _gpt4o_agent(name: str = "vision-test") -> Agent:
    return Agent(
        name=name,
        system_prompt=(
            "You are a vision assistant. Answer concisely. "
            "If you see text in the image, quote it exactly."
        ),
        llm=LLMClient(provider="openai", model="gpt-4o"),
    )


class TestOpenAIVisionGate:
    """Spec test #1, #5, #6 — real OpenAI gpt-4o vision calls."""

    def test_image_input_round_trip(self) -> None:
        require_env()
        _ensure_fixtures()
        agent = _gpt4o_agent("describe-image")
        result = agent.run(
            ["What letters do you see in this image?", Image.from_file(FIXTURES / "cat.jpg")]
        )
        assert result.output, "expected non-empty output from gpt-4o"
        letters = "".join(c for c in result.output.upper() if c.isalpha())
        assert "CAT" in letters, (
            f"vision response should mention CAT (any spacing); got: {result.output[:200]}"
        )

    def test_mixed_text_image_pdf_in_one_call(self) -> None:
        """Spec test #5 — text + image + PDF flow into one call in order."""
        require_env()
        _ensure_fixtures()
        agent = _gpt4o_agent("mixed-input")
        result = agent.run(
            [
                "I will give you an image and a contract. "
                "First report the letters in the image, then the duration "
                "mentioned in the contract. "
                "Format: 'IMAGE: <letters>; CONTRACT: <duration>'.",
                Image.from_file(FIXTURES / "cat.jpg"),
                PDF.from_file(FIXTURES / "contract.pdf"),
            ]
        )
        letters = "".join(c for c in result.output.upper() if c.isalpha())
        assert "CAT" in letters
        assert "two years" in result.output.lower() or "2 years" in result.output.lower()

    def test_pdf_text_mode_passthrough(self) -> None:
        """Spec test #2 — PDF text-mode route, exercised against gpt-3.5-turbo."""
        require_env()
        _ensure_fixtures()
        agent = Agent(
            name="pdf-text-mode",
            system_prompt="Answer concisely from the document.",
            llm=LLMClient(
                provider="openai",
                model="gpt-3.5-turbo",
                pdf_mode="text",
            ),
        )
        result = agent.run(
            [
                "What is the term length of the agreement?",
                PDF.from_file(FIXTURES / "contract.pdf"),
            ]
        )
        assert "two years" in result.output.lower() or "2 years" in result.output.lower()

    def test_tool_returning_image_flows_back_to_llm(self) -> None:
        """Spec test #6 — a tool returns an Image; the LLM reasons about it."""
        require_env()
        _ensure_fixtures()

        def take_test_image() -> Image:
            """Return a test image for the agent to inspect."""
            return Image.from_file(FIXTURES / "cat.jpg")

        agent = Agent(
            name="tool-image-return",
            system_prompt=(
                "Use the take_test_image tool to fetch the picture, "
                "then describe what letters appear in it."
            ),
            llm=LLMClient(provider="openai", model="gpt-4o"),
            tools=[FunctionTool(name="take_test_image", fn=take_test_image)],
        )
        result = agent.run("Please fetch the test image and tell me what letters it shows.")
        letters = "".join(c for c in result.output.upper() if c.isalpha())
        assert "CAT" in letters, (
            f"agent must surface the image content; got: {result.output[:200]}"
        )
