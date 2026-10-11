"""The proof scripts behind docs/durability/durability-boundaries.md still say what the page quotes.

Each script runs as a subprocess and its output is checked for the lines the page relies
on, including the one edge the page reports as open. If the SDK's behaviour changes, this
fails before the page lies.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROOFS = Path(__file__).resolve().parent.parent / "proofs"

EXPECTED = {
    "proof_1_rows.py": [
        "#2 run_end                step=run_end    status=completed",
        "#0 turn:0/tool:refund     step=hitl_pause status=interrupted",
        "run-end rows for the paused run: []",
    ],
    "proof_2_pause_resume.py": [
        "status: completed · output: 'Refund for order 1042 is on its way.'",
        "frozen in the checkpoint: {'order': '1042', 'amount': 50000, 'balance': 100}",
        "AlreadyResumed:",
        "outcomes: ['AlreadyResumed', 'AlreadyResumed', 'AlreadyResumed', 'AlreadyResumed', "
        "\"completed('Refund for o')\"]",
    ],
    "proof_3_crash.py": [
        "resume → status=completed model calls=1 tool ran with saved args → charges=[{'order': '1042', 'amount': 20}]",
        "resume → status=completed model calls=1 charges=[] (the tool was not re-run)",
        "notify ran on resume: []",
        "[no result: the agent was interrupted before this tool ran",
        # The 1.87.0 edge the page reports as open. When the resume path converts the
        # signal into a pause, this line disappears — update section 3 of the page then.
        "InterruptSignal escaped resume()",
    ],
    "proof_4_side_effects.py": [
        "plain function         status=completed charge_card fired 2×",
        "@idempotent            status=completed charge_card fired 1×",
        "called twice outside any run → fired 2 ×",
        "non-JSON-serializable value of type 'object'",
    ],
    "proof_5_run_end.py": [
        "resume → AlreadyResumed: Agent execution 'done-1' already finished (completed)",
        "last row: #2 run_end                step=run_end    status=failed",
        "resume → status: completed · charges: [{'order': '1042', 'amount': 20}]",
        "fork from the marker → ChainCheckpointError",
        "still stored: ['paused-1'] · pending: ['paused-1']",
    ],
    "proof_6_backends.py": [
        "sqlite: pause, race, claim",
        "outcomes: ['AlreadyResumed', 'AlreadyResumed', 'AlreadyResumed', 'AlreadyResumed', 'completed']",
        "un-acked rows for this run: 5",
        "after mark_synced: 0 waiting",
    ],
}


@pytest.mark.parametrize("script", sorted(EXPECTED))
def test_proof_output_matches_the_page(script: str) -> None:
    run = subprocess.run(
        [sys.executable, str(PROOFS / script)], capture_output=True, text=True, timeout=600
    )
    assert run.returncode == 0, run.stderr[-2000:]
    for line in EXPECTED[script]:
        assert line in run.stdout, f"{script}: missing {line!r}\n--- stdout ---\n{run.stdout}"
