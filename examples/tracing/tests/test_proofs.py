"""The proof scripts behind docs/tracing/trace-boundaries.md still say what the page quotes.

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
    "proof_tree.py": [
        "trace of alpha: 5 flat rows, 1 parentless root (agent.alpha)",
        "tool.lookup_alpha            runner.type=tool",
        "db.query",
        "✓ nothing of beta's run is in alpha's trace",
        "✓ nothing of alpha's run is in beta's trace",
    ],
    "proof_durable.py": [
        # Only the LLM span was on disk while the tool ran; the root was still open.
        "rows on disk while the tool was running:\n    llm.test.function-model [UNSET]\n"
        "rows on disk after the run:",
        "tool.status=error  tool.error=\"Tool 'explode' failed: carrier API returned 503\"",
        "agent.shipping            [ERROR]  exception: provider returned 503",
        "answer with an unwritable store: 'Order 1042 has shipped.'",
        "warning logged once: True",
    ],
    "proof_egress.py": [
        "exporter  agent.input = '(stripped)'",
        "exporter  gen_ai.request.messages present: False",
        "local.db  agent.input = 'Is my card [CARD] active?'",
        "exporter  gen_ai.request.messages carries the card number: False",
        "trace_id on the result: None",
        "rows in a fresh local.db: 0",
    ],
    "proof_queue.py": [
        "is_connected=True",
        "rows in local.db: 6   unsent: 6",
        "after marking 2 sent: unsent=4  rows=6",
    ],
    "proof_foreign.py": [
        "gen_ai.request.model         = 'gpt-4.1-mini'",
        "fastaiagent.runner.type      = 'llm'",
        "fastaiagent.framework        = 'openai'",
        "OpenInference keys kept: True",
        "list_traces() shows both: ['ChatCompletion', 'agent.support']",
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
