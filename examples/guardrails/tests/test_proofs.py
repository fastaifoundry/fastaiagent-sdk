"""The proof scripts behind docs/guardrails/guardrail-boundaries.md still say what the page quotes.

Each offline script runs as a subprocess and its output is checked for the lines the
page relies on. If the SDK's behaviour changes, this fails before the page lies.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROOFS = Path(__file__).resolve().parent.parent / "proofs"

EXPECTED = {
    "proof_1_positions.py": [
        "ran, in order : ['input:gate', 'input:watch-a', 'input:watch-b']",
        "firings       : [('gate', True, False, 'none'), ('watch-a', False, False, 'blocked'), "
        "('watch-b', False, True, 'blocked')]",
        "ran, in order : ['input:gate']",
        "ran, in order : ['input', 'tool_call', 'tool_result', 'output']",
    ],
    "proof_2_actions.py": [
        "mask      → output=\"The customer's SSN is [REDACTED].\"",
        "override  → output=\"I can't share that.\"",
        "reask     → output=\"The customer's SSN is on file.\"  model calls=2",
        "the model received: 'My SSN is [REDACTED], please update it.'",
        "firings: [('mask-ssn', False, False, 'masked'), ('block-ssn', True, False, 'none')]",
        "already been streamed to the caller and cannot be taken back; blocked instead",
        "passed=False errored=True action='mask' action_taken='blocked'",
    ],
    "proof_3_could_not_run.py": [
        "on_error=allow → run completed · firing=('moderation', True, True, 'none')",
        "on_error=block → GuardrailBlockedError",
        "pii             {'entities': []}                 False  True",
        "groundedness    {}                               False  True",
        "action='mask' → action_taken='blocked' errored=True",
        "cost_limit outside a run → passed=False errored=True",
    ],
    "proof_4_trace.py": [
        "run      : blocked by no-email",
        '"name": "length", "position": "output", "passed": true',
        "span detail, exported  : None",
        "dropped: ['detail']",
    ],
    "proof_5_plane_rule.py": [
        "type=regex position=tool_call blocking=True action='block' severity=high floor=True origin=plane",
        "implementation_type='code'     → None",
        "agent.to_dict()['guardrails'] → ['local-length']",
        "26/26 cases agree",
    ],
    "proof_6_roundtrip.py": [
        "no_pii() builtin         False           False          False",
        "custom fn                False           False          True",
        "action_taken=masked   → stop:",
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
