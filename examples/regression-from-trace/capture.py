"""Step 1 — run the buggy agent and capture a failing trace.

The agent uses ``lookup_order_buggy`` which returns the literal string
``"None"`` for missing orders. We pass an unknown order ID so the LLM
ingests "None" and replies with a confidently-wrong sentence — the
exact bug the rest of the loop fixes.

Run from the template directory::

    cd examples/regression-from-trace
    zsh -lc 'python capture.py'

Output:
    .fastaiagent-demo/regression-from-trace/last_trace_id.txt
        The trace ID for the next step (analyze.py).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make ``import agent`` / ``import tools`` work whether this script is
# run from inside the template dir or from the repo root.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional — env can come from the shell.

from agent import build_buggy_agent  # noqa: E402

DEMO_INPUT = "What's the status of order ORD-999?"
DEMO_DIR = Path(".fastaiagent-demo/regression-from-trace")
TRACE_ID_FILE = DEMO_DIR / "last_trace_id.txt"


def _require_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set.")
        print("Run via: zsh -lc 'python capture.py'  (from this directory)")
        raise SystemExit(0)


def main() -> int:
    _require_key()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1: Reproducing the failure with the buggy lookup_order tool…")
    agent = build_buggy_agent()
    result = agent.run(DEMO_INPUT)

    print(f"  Input:    {DEMO_INPUT!r}")
    print(f"  Output:   {result.output!r}")
    print(f"  Trace ID: {result.trace_id}")
    print()
    print(
        "  Notice the agent confidently reports delivery details for ORD-999 "
        "even though it doesn't exist — the buggy lookup_order tool silently "
        "fell back to ORD-001's record (stamped with ORD-999 as the id), and "
        "the LLM had no signal that the lookup actually failed."
    )

    assert result.trace_id, "agent.run did not produce a trace_id"
    TRACE_ID_FILE.write_text(result.trace_id)
    print()
    print(f"  Trace ID stashed at: {TRACE_ID_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
