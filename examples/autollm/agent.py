"""AutoLLM — eval-driven prompt optimization, end to end with a real LLM.

What it shows
-------------
A sentiment classifier starts with a weak, format-agnostic system prompt, so it
answers in full sentences ("The sentiment is positive.") and fails *strict*
exact-match scoring against the one-word gold label. **AutoLLM**
(``fastaiagent.optimize``) closes the loop ``harden()`` opens: it reads the dev
failures, proposes prompt rewrites, scores each on a held-out split, keeps the
best, and holdout-guards the winner so it can't overfit.

This is the OSS on-ramp story — standard prompt optimization grounded in your own
eval data, end to end in one SDK. **Real OpenAI calls, no mocks.**

The run is persisted (``persist=True``), so it shows up in ``fastaiagent ui`` →
**AutoLLM** with the full trajectory and a drill-down into each candidate's
eval run.

Run
---
    # from the repo root, so the run lands in the same local.db `fastaiagent ui` reads
    export OPENAI_API_KEY=sk-...
    python examples/autollm/agent.py
    fastaiagent ui            # open → AutoLLM → the "autollm sentiment demo" run
"""

from __future__ import annotations

import fastaiagent as fa

# A small sentiment dataset: short reviews → single-word gold labels. 18 cases is
# enough for a real, seeded train/dev/holdout split (optimize warns below ~15).
DATASET: list[dict[str, str]] = [
    {"input": "Absolutely loved this film — a total masterpiece.", "expected_output": "positive"},
    {"input": "Worst purchase I've ever made. Complete waste of money.", "expected_output": "negative"},
    {"input": "Fast shipping and the quality exceeded my expectations.", "expected_output": "positive"},
    {"input": "The app crashes every time I open it. Unusable.", "expected_output": "negative"},
    {"input": "Cozy little café with friendly staff and great coffee.", "expected_output": "positive"},
    {"input": "Rude service and the food arrived cold.", "expected_output": "negative"},
    {"input": "This book kept me hooked from the very first page.", "expected_output": "positive"},
    {"input": "Battery dies in an hour. Hugely disappointing.", "expected_output": "negative"},
    {"input": "Comfortable, well-built, and worth every penny.", "expected_output": "positive"},
    {"input": "It broke after two days and support never replied.", "expected_output": "negative"},
    {"input": "Beautiful design and incredibly easy to set up.", "expected_output": "positive"},
    {"input": "Overpriced for what you get. Would not recommend.", "expected_output": "negative"},
    {"input": "Best concert I've been to in years — unforgettable.", "expected_output": "positive"},
    {"input": "The hotel room was dirty and smelled of smoke.", "expected_output": "negative"},
    {"input": "Smooth, responsive, and a joy to use daily.", "expected_output": "positive"},
    {"input": "Misleading description; the product is nothing like the photos.", "expected_output": "negative"},
    {"input": "Delicious meal and a lovely atmosphere all evening.", "expected_output": "positive"},
    {"input": "Slow, buggy, and crashed during the demo.", "expected_output": "negative"},
]


def build_agent() -> fa.Agent:
    """A deliberately weak classifier: the prompt sets the task but no output
    *format*, so the model tends to reply in a sentence and fail exact_match."""
    return fa.Agent(
        name="sentiment-classifier",
        system_prompt="You classify the sentiment of a customer review.",
        llm=fa.LLMClient(provider="openai", model="gpt-4o-mini"),
    )


def main() -> None:
    agent = build_agent()

    print("=== AutoLLM — optimizing a sentiment classifier (real OpenAI calls) ===\n")
    report = fa.optimize(
        agent,
        DATASET,
        ["exact_match"],  # strict: output.strip() == gold — rewards format discipline
        config=fa.OptimizeConfig(
            levers=("instructions",),       # tune the system prompt (P1 lever)
            max_iterations=3,
            candidates_per_iteration=2,
            patience=2,
            seed=0,
        ),
        run_name="autollm sentiment demo",
        persist=True,                       # land the run in local.db for the UI
    )

    print(report.summary())
    print()
    print(f"baseline dev = {report.baseline.score:.3f}    best dev = {report.best.score:.3f}")
    print("→ improved" if report.improved else "→ no improvement this run")

    if report.run_id:
        print(
            f"\nRun persisted as {report.run_id[:8]} — open `fastaiagent ui` → AutoLLM "
            "to see the trajectory and drill into each candidate's eval run."
        )

    # End-to-end: apply the winning prompt and probe a fresh review.
    print("\n=== winning system prompt ===")
    print(report.best_candidate.system_prompt or "(unchanged from baseline)")

    tuned = report.apply_to(agent)
    probe = "The screen flickers constantly and tech support was useless."
    print("\n=== tuned agent on a held-out probe ===")
    print(f"review: {probe}")
    print(f"label : {tuned.run(probe).output!r}")


if __name__ == "__main__":
    main()
