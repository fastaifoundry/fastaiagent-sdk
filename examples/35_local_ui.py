"""Example 35 — Local UI end-to-end.

Produces a representative slice of local.db content (traces, a guardrail event,
an eval run) so you can open `fastaiagent ui` and see every surface populated.

Prereqs:
    pip install 'fastaiagent[ui,openai]'
    export OPENAI_API_KEY=...

Run:
    python examples/35_local_ui.py

Then:
    fastaiagent ui
"""

from __future__ import annotations

import os

# Turn on guardrail event logging so the UI's Guardrails page has data.
os.environ.setdefault("FASTAIAGENT_UI_ENABLED", "true")

from fastaiagent import Agent, LLMClient  # noqa: E402
from fastaiagent.eval import evaluate  # noqa: E402
from fastaiagent.guardrail.builtins import no_pii  # noqa: E402
from fastaiagent.prompt import PromptRegistry  # noqa: E402


def _build_agent() -> Agent:
    """A small assistant guarded against leaking PII in its responses."""
    llm = LLMClient(provider="openai", model="gpt-4o-mini")
    registry = PromptRegistry()
    registry.register(
        "ui-example.assistant",
        template=(
            "You are a concise customer-support assistant. "
            "Answer in under 40 words. Never include phone numbers, SSNs, or credit cards."
        ),
    )
    prompt = registry.load("ui-example.assistant")
    return Agent(
        name="ui-example-assistant",
        system_prompt=prompt.template,
        llm=llm,
        guardrails=[no_pii()],
    )


def _produce_traces(agent: Agent) -> None:
    """Two runs — one clean, one that exercises the no_pii guardrail."""
    print("► running agent on a clean query")
    clean = agent.run("What's a good first step when a customer asks for a refund?")
    print(f"  trace: {clean.trace_id}")
    print(f"  output: {clean.output[:120]}")

    print("\n► running agent on a query with PII (SSN)")
    try:
        guarded = agent.run(
            "Hi — my SSN is 111-22-3333 and I need a refund on my last order."
        )
        print(f"  trace: {guarded.trace_id}")
        print(f"  output: {guarded.output[:120]}")
    except Exception as exc:  # noqa: BLE001
        print(f"  guardrail blocked: {exc}")


def _run_eval(agent: Agent) -> None:
    """A mini eval run so /evals has a row."""
    print("\n► running eval — 3 cases, exact_match scorer")
    dataset = [
        {
            "input": "Say 'hello'.",
            "expected_output": "hello",
        },
        {
            "input": "Reply with the word 'yes'.",
            "expected_output": "yes",
        },
        {
            "input": "Respond with the word 'no'.",
            "expected_output": "no",
        },
    ]
    results = evaluate(
        agent_fn=agent.run,
        dataset=dataset,
        scorers=["exact_match"],
        run_name="ui-example-smoke",
        dataset_name="ui-example.jsonl",
        agent_name="ui-example-assistant",
    )
    print(f"  {results.summary()}")


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required to run this example.")

    agent = _build_agent()
    _produce_traces(agent)
    _run_eval(agent)

    print()
    print("Done. Now open the UI:")
    print("    fastaiagent ui")
    print()
    print("You'll see:")
    print("  • /         - the runs summarized on Overview")
    print("  • /traces   - two traces with spans, tokens, cost")
    print("  • /guardrails - the no_pii event on the PII-laced query")
    print("  • /prompts  - ui-example.assistant v1 (click to edit)")
    print("  • /evals    - the ui-example-smoke run + its cases")
    print("  • /agents   - the ui-example-assistant agent card")


if __name__ == "__main__":
    main()
