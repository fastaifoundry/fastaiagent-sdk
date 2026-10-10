"""Step 4 — score what's live: v1 against the labelled dataset.

A persisted eval run, so it shows in the UI and becomes the baseline the CI gate
(step 7) compares against. Each case links to its trace, and the prompt's
lineage panel now lists this eval run too.

UI: Eval Runs → "ticket-triage v1".
"""

from __future__ import annotations

from collections import Counter

from triage import AGENT, TriageMatch, current_version, dataset_path, load_agent, save_json

from fastaiagent.eval import Dataset, evaluate


def main() -> None:
    live = current_version()
    dataset = Dataset.from_jsonl(dataset_path())
    run_name = f"ticket-triage v{live.version}"

    results = evaluate(
        load_agent(alias="production").arun,
        dataset,
        [TriageMatch()],
        run_name=run_name,
        agent_name=AGENT,
        persist=True,
    )
    print(results.summary())

    misses = Counter(r.reason for r in results.scores["triage_match"] if not r.passed)
    print("\nwhat it gets wrong:")
    for miss, n in misses.most_common(8):
        print(f"  {n:>2}  {miss}")

    save_json("baseline.json", {"run_name": run_name, "run_id": results.run_id})
    print(f"\nbaseline eval run: {run_name!r} ({results.run_id})")


if __name__ == "__main__":
    main()
