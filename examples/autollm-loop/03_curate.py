"""Step 3 — the traffic becomes a dataset: curate the traces, then label them.

``curate_from_traces`` turns every captured run into a case — the ticket, what
the agent answered, and the trace it came from — flagged ``needs_review``: a
person decides what the right answer was. In real life your support leads do
that in the UI's Dataset Editor. Here the labels come from
``data/tickets.jsonl``, so the loop runs unattended.

The file lands where the Dataset Editor reads it, so open it, change a label,
and the next step uses your change.

UI: Datasets → ticket-triage.
"""

from __future__ import annotations

from triage import AGENT, TriageMatch, dataset_path, label_text, tickets

from fastaiagent.eval import Dataset, curate_from_traces


def main() -> None:
    labels = {row["ticket"]: label_text(row["label"]) for row in tickets()}
    match = TriageMatch()
    curated = curate_from_traces(
        agent=AGENT, filter="all", mark_output_as_expected=False, dedup_by="input", limit=500
    )
    print(f"curated: {curated.coverage_summary()}")

    cases, wrong = [], 0
    for item in curated:
        expected = labels.get(item["input"])
        if expected is None:
            continue  # a run that isn't one of the tickets
        wrong += not match.score(item["input"], item.get("actual_output", ""), expected).passed
        cases.append(
            {
                "input": item["input"],
                "expected_output": expected,
                "source_trace_id": item["source_trace_id"],
            }
        )

    path = Dataset(cases).to_jsonl(dataset_path())
    print(f"labelled {len(cases)} cases → {path}")
    print(f"production got {wrong} of {len(cases)} wrong by the support leads' labels.")


if __name__ == "__main__":
    main()
