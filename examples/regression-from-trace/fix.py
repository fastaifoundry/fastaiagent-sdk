"""Step 3 — fork the failing trace and swap in the fixed tool.

Loads the trace from ``last_trace_id.txt``, forks at the root step,
calls :py:meth:`ForkedReplay.with_tool_override` to substitute the
fixed ``lookup_order`` (keeping the LLM, prompt, and every other tool
identical), and reruns LIVE so the LLM sees the corrected tool output
and produces a sane reply.

Why "live" rerun and not ``determinism="recorded"``: the original
LLM response is itself wrong — it parroted the buggy tool output. To
get a corrected reply we need the LLM to actually re-read the new
tool output and re-generate. ``determinism="recorded"`` is for
locking *prompt* fixes, not tool fixes.

Run from the template directory::

    cd examples/regression-from-trace
    zsh -lc 'python fix.py'

Output:
    .fastaiagent-demo/regression-from-trace/fixed_output.txt
        The fixed reply, consumed by save_test.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from tools import fixed_lookup_order_tool  # noqa: E402

from fastaiagent.trace.replay import Replay  # noqa: E402

DEMO_DIR = Path(".fastaiagent-demo/regression-from-trace")
TRACE_ID_FILE = DEMO_DIR / "last_trace_id.txt"
FIXED_OUTPUT_FILE = DEMO_DIR / "fixed_output.txt"


def _require_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set.")
        raise SystemExit(0)


def main() -> int:
    if not TRACE_ID_FILE.exists():
        print(f"No trace ID at {TRACE_ID_FILE}. Run capture.py first.")
        return 1
    _require_key()

    trace_id = TRACE_ID_FILE.read_text().strip()
    print(f"Step 3: Forking trace {trace_id} and swapping in the fixed tool…")

    replay = Replay.load(trace_id)
    forked = replay.fork_at(step=0).with_tool_override("lookup_order", fixed_lookup_order_tool())
    rerun = forked.rerun()

    print()
    print(f"  Original (buggy) output: {rerun.original_output!r}")
    print(f"  Rerun (fixed) output:    {rerun.new_output!r}")
    print(f"  Rerun trace ID:          {rerun.trace_id}")

    FIXED_OUTPUT_FILE.write_text(str(rerun.new_output))
    print()
    print(f"  Fixed output stashed at: {FIXED_OUTPUT_FILE}")
    print("  (save_test.py picks this up as the expected_output for the regression case.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
