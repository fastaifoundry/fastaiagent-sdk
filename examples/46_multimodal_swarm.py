"""Example 46: Multimodal — insurance-claim Swarm.

A 3-agent swarm receives a multimodal claim payload (text + photo + PDF
policy) and routes it: ``triage`` reads the request and hands off to
``photo_assessor`` (vision model) or ``policy_agent`` (PDF reader). The
swarm input is the same ``list[ContentPart]`` that ``Agent.run`` accepts.

Usage::

    zsh -lc 'export OPENAI_API_KEY=sk-...'
    zsh -lc 'python examples/46_multimodal_swarm.py'

Prefers OpenAI (always vision-capable on gpt-4o); falls back to skipping
when the key isn't set rather than failing.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastaiagent import PDF, Agent, Image, LLMClient, Swarm

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "multimodal"


def _ensure_fixtures() -> tuple[Path, Path]:
    cat = FIXTURES / "cat.jpg"
    pdf = FIXTURES / "contract.pdf"
    if not cat.exists() or not pdf.exists():
        from tests.fixtures.multimodal._make_fixtures import main as make

        make()
    return cat, pdf


def build_swarm() -> Swarm:
    llm = LLMClient(provider="openai", model="gpt-4o")

    triage = Agent(
        name="triage",
        system_prompt=(
            "You are a claim triage agent. The user has filed a claim that "
            "includes a photo and a policy document. Decide which specialist "
            "should look at it next. If the photo shows damage, hand off to "
            "photo_assessor. If the user is asking a coverage question, hand "
            "off to policy_agent. Do not answer the claim yourself."
        ),
        llm=llm,
    )
    photo_assessor = Agent(
        name="photo_assessor",
        system_prompt=(
            "You are a damage-assessment specialist. Describe what you see "
            "in the photo. Quote any text it shows verbatim. Be concise."
        ),
        llm=llm,
    )
    policy_agent = Agent(
        name="policy_agent",
        system_prompt=(
            "You are a policy reader. Answer policy / coverage / term-length "
            "questions strictly from the document the user provided. "
            "Be concise."
        ),
        llm=llm,
    )

    return Swarm(
        name="claims_swarm",
        agents=[triage, photo_assessor, policy_agent],
        entrypoint="triage",
        handoffs={
            "triage": ["photo_assessor", "policy_agent"],
            "photo_assessor": [],
            "policy_agent": [],
        },
        max_handoffs=2,
    )


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set")
        print("Run: zsh -lc 'python examples/46_multimodal_swarm.py'")
        return

    cat_path, pdf_path = _ensure_fixtures()
    img = Image.from_file(cat_path)
    pdf = PDF.from_file(pdf_path)

    swarm = build_swarm()

    # Request 1 — visual question; triage hands off to photo_assessor.
    print("--- Request 1: photo damage ---")
    result1 = swarm.run(
        [
            "I'm filing a damage claim. What letters appear on the object in this photo?",
            img,
            pdf,
        ]
    )
    print(f"Output: {result1.output}")
    handoffs1 = [
        c for c in result1.tool_calls if str(c.get("tool_name", "")).startswith("handoff_to_")
    ]
    print(f"Handoffs: {[c.get('tool_name') for c in handoffs1]}")
    print(f"Trace ID: {result1.trace_id}")
    print()

    # Request 2 — coverage question; triage hands off to policy_agent.
    print("--- Request 2: policy coverage ---")
    result2 = swarm.run(
        [
            "What is the term length stated in the attached policy?",
            img,
            pdf,
        ]
    )
    print(f"Output: {result2.output}")
    handoffs2 = [
        c for c in result2.tool_calls if str(c.get("tool_name", "")).startswith("handoff_to_")
    ]
    print(f"Handoffs: {[c.get('tool_name') for c in handoffs2]}")
    print(f"Trace ID: {result2.trace_id}")

    print(
        "\nTo render the swarm topology in the Local UI, register the "
        "Swarm with build_app:\n"
        "    from fastaiagent.ui.server import build_app\n"
        "    app = build_app(runners=[swarm])\n"
        "Then visit http://127.0.0.1:7843/workflows/swarm/claims_swarm"
    )


if __name__ == "__main__":
    main()
