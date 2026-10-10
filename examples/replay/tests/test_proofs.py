"""The proof scripts behind docs/replay/replay-boundaries.md still say what the page quotes.

Each script runs as a subprocess, offline, against a throwaway local.db, and its
output is checked for the lines the page relies on. If the SDK's behaviour
changes, this fails before the page lies.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROOFS = Path(__file__).resolve().parent.parent / "proofs"

EXPECTED = {
    "proof_blueprint.py": [
        "api_key in agent.llm.config: False",
        "[3] tool.lookup_order",
        "the live function: True   replay_class=read_only",
        "guardrails  ['no_secrets']",
        "memory      None   (the original's window holds 2 messages)",
    ],
    "proof_recorded.py": [
        "byte-identical: True",
        "model calls during the rerun: 0   tool calls: 1",
        "llm.test.function-model    replay.mode=recorded",
        "compare(): status=ok  diverged_at=None",
        "prompt the rerun was given: 'Répondez en français.'",
        "answer: 'Order 1042 was shipped via DHL on 12 September.'",
    ],
    "proof_tools.py": [
        "replay_class: charge_card='side_effecting'  GET carrier_status='side_effecting'",
        "recorded rerun, same process: 'Done: charged 42.00.'   charges=[42.0, 42.0]",
        "tool.status = error | tool.error = 'No function attached to this tool'",
        "with_tool_override: 'Done: charged 42.00.'   charges=[42.0, 42.0]",
    ],
    "proof_miss.py": [
        "captured model responses: 1",
        "on_miss='error' → ReplayError: determinism='recorded' ran out of captured LLM responses",
        "falling through to a LIVE test call",
        "the live call to provider 'test' failed: LLMError",
    ],
    "proof_isolation.py": [
        "rerun:        prompt roles ['system', 'user']   memory holds 4 messages",
        "'guardrail.no_secrets']",
        "A replay rebuilds the agent without a checkpointer, so it cannot hold a pause",
    ],
    "proof_regression.py": [
        "source_trace_id",
        "evaluate() on the saved case: exact_match passed=True score=1.0",
        "evaluate() with the old prompt:  exact_match passed=False score=0.0",
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
