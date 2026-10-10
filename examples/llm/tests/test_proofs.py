"""The proof scripts behind docs/llm/llm-boundaries.md still say what the page quotes.

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
    "proof_1_wire.py": [
        "keys    : ['max_completion_tokens', 'messages', 'model', 'tools']",
        "keys    : ['max_tokens', 'messages', 'model', 'tools']",
        "keys    : ['max_tokens', 'messages', 'model', 'system', 'tools']",
        '"type": "tool_result", "tool_use_id": "call_1"',
        '"input_schema"',
    ],
    "proof_3_presets.py": [
        "provider keys in total: 19",
        "response_format in body: False · parallel_tool_calls in body: False · max_tokens key: max_tokens",
        "You must respond with valid JSON matching this schema ('City')",
    ],
    "proof_6_numbers.py": [
        "gpt-4o-mini-2024-07-18   1M in + 1M out → 0.75",
        "my-private-finetune      1M in + 1M out → None",
        "max_retries=2 : 3 requests, content='recorded'",
        "max_retries=0 : LLMProviderError status=429",
        '"size_bytes": 68',
        "dropped on export: ['gen_ai.request.messages', 'gen_ai.response.content']",
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
