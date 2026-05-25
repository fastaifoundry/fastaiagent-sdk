"""Step 2 — load the captured trace and surface the failing step.

Reads ``last_trace_id.txt`` from capture.py, loads the trace, prints
each step's name + a short attribute summary so the user can spot the
tool node where ``lookup_order`` returned the bogus ``"None"`` string.

Run from the template directory::

    cd examples/regression-from-trace
    python analyze.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fastaiagent.trace.replay import Replay  # noqa: E402

DEMO_DIR = Path(".fastaiagent-demo/regression-from-trace")
TRACE_ID_FILE = DEMO_DIR / "last_trace_id.txt"


def main() -> int:
    if not TRACE_ID_FILE.exists():
        print(f"No trace ID at {TRACE_ID_FILE}. Run capture.py first to produce a failing trace.")
        return 1

    trace_id = TRACE_ID_FILE.read_text().strip()
    print(f"Step 2: Loading trace {trace_id}…")
    replay = Replay.load(trace_id)

    print()
    print(replay.summary())
    print()
    print("Steps with attributes:")
    for step in replay.step_through():
        # Look for the tool call result — it's the smoking gun.
        attrs = step.attributes or {}
        interesting_keys = (
            "tool.name",
            "tool.input",
            "tool.output",
            "agent.output",
            "gen_ai.response.content",
        )
        snippet = {k: attrs[k] for k in interesting_keys if k in attrs}
        print(f"  [{step.step}] {step.span_name}")
        for k, v in snippet.items():
            shown = str(v)
            if len(shown) > 120:
                shown = shown[:117] + "…"
            print(f"        {k} = {shown}")

    print()
    print("Look at the tool.* step — for unknown order IDs the buggy")
    print("lookup_order silently fell back to ORD-001's record (with the")
    print("requested id stamped on) so the LLM had nothing to flag as")
    print("wrong. fix.py overrides the tool with a fixed implementation")
    print("that returns {'error': ...} for missing IDs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
