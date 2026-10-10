"""The proof scripts behind docs/evaluation/autollm-how-it-works.md still say what the page quotes.

Each script runs as a subprocess, offline, and its output is checked for the lines
the page relies on. If the SDK's behaviour changes, this fails before the page lies.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROOFS = Path(__file__).resolve().parent.parent / "proofs"

EXPECTED = {
    "proof_isolation.py": [
        "original prompt : 'You triage tickets.'",
        "original memory : None",
        "demo in prompt  : True",
        "customer two saw: ['customer two: hello']",
    ],
    "proof_errors.py": [
        "evaluate() pass-rate 1.000  |  AutoLLM score 0.500  (3 errored of 6)",
        "evaluate() pass-rate 0.667  |  AutoLLM score 0.667  (0 errored of 6)",
    ],
    "proof_judges.py": [
        "audit_judge is named 'g_eval', like a different scorer in scorers",
        "audit_judge is None; falling back to selection_judge",
    ],
    "proof_budget.py": [
        "can't cover the baseline and the holdout guard",
        "(stopped: budget)",
        "evaluation passes written to local.db: 5  (cap was 5)",
    ],
}


@pytest.mark.parametrize("script", sorted(EXPECTED))
def test_proof_output_matches_the_page(script: str) -> None:
    run = subprocess.run(
        [sys.executable, str(PROOFS / script)], capture_output=True, text=True, timeout=300
    )
    assert run.returncode == 0, run.stderr[-2000:]
    for line in EXPECTED[script]:
        assert line in run.stdout, f"{script}: missing {line!r}\n--- stdout ---\n{run.stdout}"
