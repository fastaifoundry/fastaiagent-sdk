"""The proof scripts behind docs/knowledge-base/kb-boundaries.md still say what the page quotes.

Each offline script runs as a subprocess and its output is checked for the lines the
page relies on. If the SDK's behaviour changes — including the two defects the page
reports as they are — this fails before the page lies.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROOFS = Path(__file__).resolve().parent.parent / "proofs"

EXPECTED = {
    "proof_1_chunks.py": [
        "characters shared between neighbours: [-2, -2, -2, -2]",
        "'within 30 days … full refund' lives in chunk [0]",
    ],
    "proof_2_embedder.py": [
        "Embedding dimension mismatch: stored=384, current embedder=128",
        "4 chunks back, 1 text embedded (a one-word dimension probe)",
    ],
    "proof_3_matchers.py": [
        "identical: True",
        "top hybrid score 1.000",
    ],
    "proof_5_restart.py": [
        "chunks=3 embedded_on_open=1 (the probe)",
        "deleted 1 chunk(s) from errors.md",
        "chunks=2 embedded_on_open=1 (the probe)",
    ],
    "proof_6_backends.py": [
        "faiss   {'same': 1.0, 'orthogonal': 0.0, 'opposite': -1.0, 'cos=0.6': 0.6}",
        "chroma  {'same': 1.0, 'orthogonal': 0.0, 'opposite': -1.0, 'cos=0.6': 0.6}",
        "qdrant  {'same': 1.0, 'orthogonal': 0.0, 'opposite': -1.0, 'cos=0.6': 0.6}",
        # The 1.87.0 defect the page reports. When the backend converts the vectors
        # itself this line disappears — update section 6 of the page then.
        "chroma  ValueError from the backend",
        "chroma  ChromaVectorStore top before delete: refunds.md · removed 1 · top after: errors.md",
    ],
}
NEEDS = {
    "proof_2_embedder.py": ["fastembed"],
    "proof_3_matchers.py": ["fastembed"],
    "proof_5_restart.py": ["fastembed"],
    "proof_6_backends.py": ["fastembed", "chromadb", "qdrant_client"],
}


@pytest.mark.parametrize("script", sorted(EXPECTED))
def test_proof_output_matches_the_page(script: str) -> None:
    for module in NEEDS.get(script, []):
        pytest.importorskip(module)
    run = subprocess.run(
        [sys.executable, str(PROOFS / script)], capture_output=True, text=True, timeout=600
    )
    assert run.returncode == 0, run.stderr[-2000:]
    for line in EXPECTED[script]:
        assert line in run.stdout, f"{script}: missing {line!r}\n--- stdout ---\n{run.stdout}"
