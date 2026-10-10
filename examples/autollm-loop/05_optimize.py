"""Step 5 — AutoLLM: let the labelled dataset write the next prompt.

``optimize()`` splits the dataset (train / dev / holdout, seeded), reads the
train failures — each with the label the support leads gave and the scorer's
reason — and proposes rewrites of the prompt (the ``instructions`` lever) and
worked examples (the ``fewshot`` lever). Every candidate is a real, persisted
eval run on dev; the best is kept only if it beats the current best. Finally
the winner faces the holdout, which no step of the search ever saw, and is
thrown away if it does worse there than v1.

Nothing is shipped here: step 6 registers the winner as a new version.

UI: AutoLLM → the run (trajectory, the Winner card next to v1, every
candidate's eval run).
"""

from __future__ import annotations

from triage import (
    PROPOSER_MODEL,
    TriageMatch,
    current_version,
    dataset_path,
    load_agent,
    save_json,
)

import fastaiagent as fa
from fastaiagent.eval import Dataset


def main() -> None:
    live = current_version()
    report = fa.optimize(
        load_agent(alias="production"),
        Dataset.from_jsonl(dataset_path()),
        [TriageMatch()],
        config=fa.OptimizeConfig(
            levers=("instructions", "fewshot"),
            max_iterations=6,
            patience=2,
            candidates_per_iteration=3,
            seed=0,
            max_eval_runs=40,  # a hard cap on evaluation passes, the holdout guard included
        ),
        proposer_llm=fa.LLMClient(provider="openai", model=PROPOSER_MODEL),
        run_name=f"ticket-triage from v{live.version}",
    )
    print(report.summary())

    save_json(
        "optimize.json",
        {
            "run_id": report.run_id,
            "from_version": live.version,
            "improved": report.improved,
            "reverted": report.reverted,
            "dev": {"baseline": report.baseline.score, "best": report.best.score},
            "holdout": {
                "baseline": report.holdout_baseline.score if report.holdout_baseline else None,
                "best": report.holdout_best.score if report.holdout_best else None,
            },
            "winner": report.best_candidate.to_dict(),
        },
    )
    verdict = "a winner to promote" if report.improved else "nothing beat v1 — production stays"
    print(f"\n{verdict} (AutoLLM run {report.run_id})")


if __name__ == "__main__":
    main()
