"""The proof scripts behind docs/prompts/prompt-boundaries.md still say what the page quotes.

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
    "proof_1_placeholders.py": [
        "variables: ['company', 'name', 'tone']",
        "left in the text, verbatim : ['{{tone}}', '{{@legal}}']",
    ],
    "proof_2_versions.py": [
        "production still v 1",
        "v1 now reads           : 'REPLACED {{name}}'",
        "'latest_version': 1, 'versions': 3",
    ],
    "proof_3_fragments.py": [
        "diff(1, 1)     : (no template changes)",
        "prompt.version=1  system prompt sent: ['You help customers. Be formal.']",
        "prompt.version=1  system prompt sent: ['You help customers. Be casual and brief.']",
    ],
    "proof_4_lineage.py": [
        "run()    : ['triage v2']",
        "astream(): ['triage v2']",
        "run()    : ['(no prompt)']",
        "llm spans, in order: ['triage v2', 'summarizer v1', 'triage v2']",
    ],
    "proof_6_plane_side.py": [
        "registry B, another folder: Prompt 'support-prompt' not found",
        "prompt_slug=   → prompt_slug: support-prompt · system_prompt: ''",
        "'prompt.environment': 'production'",
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
