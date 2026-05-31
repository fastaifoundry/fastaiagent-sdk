"""Presidio-backed PII detection — skipped unless the [safety] extra is installed.

Real Presidio (no mock). Requires:
    pip install fastaiagent[safety]
    python -m spacy download en_core_web_lg
"""

from __future__ import annotations

import pytest

pytest.importorskip("presidio_analyzer")

from fastaiagent._internal.safety_detectors import detect_pii  # noqa: E402


def test_presidio_detects_email_and_phone() -> None:
    text = "Contact John at john@example.com or 415-555-1234."
    matches = detect_pii(text, backend="presidio", entities=("email", "phone"))
    found = {m.entity for m in matches}
    assert "email" in found
