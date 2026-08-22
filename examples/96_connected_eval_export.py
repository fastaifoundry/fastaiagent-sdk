"""Example 96: Agent-CI verdicts as governance evidence (connected mode).

Agent CI gates your build locally (see 95_agent_ci_gate.py). Connect to an
Enterprise control plane and each gated run's **verdict** is also reported, so
the org can see which agents produce eval evidence — and treat an agent that
produces none as a governance finding rather than a blank space.

Run it:

    export FASTAIAGENT_API_KEY=...      # needs the eval:execute scope
    export FASTAIAGENT_TARGET=https://your-plane.example.com
    python examples/96_connected_eval_export.py

Without a plane this still runs — the export path is a strict no-op when
disconnected, and the local gate behaves exactly as it always does.

What travels: run aggregates, the gate outcome, the thresholds demanded, git
provenance, and per-case scorer verdicts + trace_ids.
What never travels: case inputs, expected outputs, actual outputs. The plane
joins content through trace_id against traces it already ingested.

See docs/platform/connected-eval-export.md for the full wire protocol.
"""

import json
import os

import fastaiagent as fa
from fastaiagent import Agent
from fastaiagent.eval import evaluate, gate
from fastaiagent.eval.platform_export import build_payloads, eval_export_enabled
from fastaiagent.eval.results import record_gate_result
from fastaiagent.testing import TestModel

DATASET = [
    {"input": "refund window", "expected_output": "30 days"},
    {"input": "shipping time", "expected_output": "2 business days"},
]


def main() -> None:
    # 1. Connect (optional). Everything below works disconnected too.
    api_key = os.environ.get("FASTAIAGENT_API_KEY")
    if api_key:
        fa.connect(
            api_key=api_key,
            target=os.environ.get("FASTAIAGENT_TARGET", "https://app.fastaiagent.net"),
            # export_evals=False,   # ← the opt-out. Honored always; the plane
            #                          cannot override it (same rule as traces).
        )
    print(f"connected={fa.is_connected}  export_evals={eval_export_enabled()}")

    # 2. A normal gated run. TestModel keeps the example deterministic; swap in
    #    LLMClient(provider="openai", ...) for the real thing.
    support = Agent(
        name="northwind-support",  # ← this name is what the plane attributes evidence to
        llm=TestModel(response="30 days"),
    )
    results = evaluate(
        support.run, DATASET, scorers=["contains"], run_name="example-96", concurrency=2
    )
    report = gate(results, fail_under=["overall.pass_rate=0.9"], max_error_rate=0.1)
    print(f"gate: {report.outcome}")
    for line in report.describe():
        print(f"  {line}")

    # 3. Recording the verdict is what makes the run *evidence* — the plane
    #    requires a gate outcome, and the gate necessarily runs after persistence.
    #    The pytest plugin and `fastaiagent eval run` do this for you; this is the
    #    explicit form for a hand-rolled harness.
    if results.run_id:
        record_gate_result(
            results.run_id,
            gate_outcome=report.outcome,
            thresholds={"overall.pass_rate": 0.9},
        )

    # 4. Inspect exactly what would leave the machine. Same as
    #    `fastaiagent eval export --dry-run` — hand this to a security review.
    payload = build_payloads(run_id=results.run_id, limit=1)
    print("\n--- wire payload (metadata only) ---")
    print(json.dumps({"runs": payload}, indent=2)[:900])

    banned = {"input", "expected_output", "actual_output"}
    for run in payload:
        assert not (banned & set(run))
        for case in run["cases"]:
            assert not (banned & set(case))
    print("\n✓ no case content on the wire — the plane joins it via trace_id")

    if fa.is_connected:
        fa.disconnect()  # flushes any queued verdicts on the way out


if __name__ == "__main__":
    main()
