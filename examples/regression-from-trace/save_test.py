"""Step 4 — append the rerun as a regression case.

Reads the fixed output from ``fix.py`` and writes a JSONL line whose
field names match what ``fastaiagent.eval.evaluate(...)`` expects, so
the same file is both the test artifact for this template *and* a
real dataset future eval runs can read.

``rerun.save_as_test()`` does the JSONL write — we re-construct a
ReplayResult-shaped value from the staged outputs because save_test
is meant to be runnable standalone (no need to keep a Python object
in memory across the five steps).

Run from the template directory::

    cd examples/regression-from-trace
    python save_test.py

Output:
    regression_dataset.jsonl
        Appended with one row: ``{"input": ..., "expected_output": ...,
        "trace_id": ..., "created_at": ...}``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fastaiagent.trace.replay import ReplayResult  # noqa: E402

DEMO_DIR = Path(".fastaiagent-demo/regression-from-trace")
TRACE_ID_FILE = DEMO_DIR / "last_trace_id.txt"
FIXED_OUTPUT_FILE = DEMO_DIR / "fixed_output.txt"
DATASET = _HERE / "regression_dataset.jsonl"
DEMO_INPUT = "What's the status of order ORD-999?"


def main() -> int:
    if not FIXED_OUTPUT_FILE.exists() or not TRACE_ID_FILE.exists():
        print("Missing fix.py output — run capture.py then fix.py first.")
        return 1

    expected = FIXED_OUTPUT_FILE.read_text().strip()
    source_trace_id = TRACE_ID_FILE.read_text().strip()
    print(f"Step 4: Saving regression case to {DATASET.name}…")

    # ``save_as_test`` is a method on ReplayResult so we wrap the staged
    # values in one. This lets the template walk through the loop step
    # by step without keeping a Python object across processes.
    result = ReplayResult(
        original_output="(captured separately — see last_trace_id.txt)",
        new_output=expected,
        steps_executed=1,
        trace_id=source_trace_id,
    )
    result.save_as_test(
        DATASET,
        input=DEMO_INPUT,
        expected_output=expected,
        source_trace_id=source_trace_id,
    )

    line_count = sum(1 for _ in DATASET.open())
    print(f"  Dataset now has {line_count} case(s) at: {DATASET}")
    print()
    print(
        "  Each line is a JSON record with ``input`` and ``expected_output`` "
        "fields — that's what ``evaluate()`` consumes in verify.py."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
