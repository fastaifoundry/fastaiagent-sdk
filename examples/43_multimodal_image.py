"""Example 43: Multimodal — describe an image with a vision LLM.

Sends a synthetic test image (the letters "CAT" rendered onto a 200x200
JPEG) to OpenAI gpt-4o and Anthropic Claude Sonnet 4. The same Python
code works for both providers — provider-specific wire formatting is
hidden inside ``LLMClient``.

Usage::

    zsh -lc 'export OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-...'
    zsh -lc 'python examples/43_multimodal_image.py'

The fixture image is regenerated on demand via ``tests/fixtures/multimodal/_make_fixtures.py``.

After running, open the trace in ``fastaiagent ui`` — the Input tab on
the LLM span renders the image inline next to the prompt. See
``docs/ui/multimodal.md`` and the screenshot at
``docs/ui/screenshots/sprint1-2-multimodal-input.png`` for a visual
walkthrough.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastaiagent import Agent, Image, LLMClient

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "multimodal"


def _ensure_fixture() -> Path:
    """Generate ``cat.jpg`` if a fresh checkout is missing it."""
    cat = FIXTURES / "cat.jpg"
    if not cat.exists():
        from tests.fixtures.multimodal._make_fixtures import main as make

        make()
    return cat


def run_with(provider: str, model: str, env_var: str) -> None:
    if not os.environ.get(env_var):
        print(f"Skipping {provider}: {env_var} not set")
        return

    cat = _ensure_fixture()
    agent = Agent(
        name=f"vision-{provider}",
        system_prompt=(
            "You are a vision assistant. Be concise. "
            "If you see text in the image, quote it exactly."
        ),
        llm=LLMClient(provider=provider, model=model),
    )
    result = agent.run(
        ["What letters appear in this image?", Image.from_file(cat)]
    )
    print(f"--- {provider} / {model} ---")
    print(f"Output: {result.output}")
    print(f"Tokens used: {result.tokens_used}")
    print(f"Trace ID:    {result.trace_id}")
    print()


if __name__ == "__main__":
    run_with("openai", "gpt-4o", "OPENAI_API_KEY")
    run_with("anthropic", "claude-sonnet-4-20250514", "ANTHROPIC_API_KEY")
