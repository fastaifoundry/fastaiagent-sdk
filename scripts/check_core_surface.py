#!/usr/bin/env python3
"""Assert the library-usage surface works with *only* the core install.

Run inside a venv built with ``pip install .`` — no extras, no ``[dev]``::

    python scripts/check_core_surface.py

The success criterion this encodes: ``pip install fastaiagent`` inside someone
else's project (a LangChain app that just wants to borrow ``run_guardrail``)
should be as unremarkable as ``pip install jsonschema``. If borrowing a
primitive silently requires an extra, that is a packaging regression, and it is
the kind that is invisible in the normal test matrix — which installs ``[dev]``
and therefore has everything.

Deliberately plain Python, not pytest: pytest is not installed in this venv,
and adding it would defeat the point.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys

failures: list[str] = []


def check(label: str) -> _Check:
    return _Check(label)


class _Check:
    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self) -> _Check:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> bool:
        if exc is None:
            print(f"  ok    {self.label}")
        else:
            print(f"  FAIL  {self.label}: {exc!r}")
            failures.append(f"{self.label}: {exc!r}")
        return True  # keep going; report everything in one run


print("core import surface (no extras installed)")

with check("import fastaiagent"):
    import fastaiagent

    assert fastaiagent.__version__, "missing __version__"

with check("from fastaiagent import run_guardrail, plane_guardrails_for_agent"):
    from fastaiagent import Guardrail, plane_guardrails_for_agent, run_guardrail

    # GuardrailType lives on the submodule, not the top-level namespace.
    from fastaiagent.guardrail import GuardrailType

with check("plane_guardrails_for_agent returns [] when not connected"):
    assert plane_guardrails_for_agent("some-agent") == []

with check("regex guardrail blocks a match end to end"):
    blocker = Guardrail(
        name="no-ssn",
        guardrail_type=GuardrailType.regex,
        config={"pattern": r"\d{3}-\d{2}-\d{4}", "should_match": False},
    )
    blocked = asyncio.run(run_guardrail(blocker, "my ssn is 123-45-6789"))
    assert not blocked.passed, f"expected block, got {blocked!r}"
    allowed = asyncio.run(run_guardrail(blocker, "nothing sensitive here"))
    assert allowed.passed, f"expected pass, got {allowed!r}"
    assert not allowed.errored, "guardrail errored instead of evaluating"

with check("code guardrail runs via the sync Guardrail.execute path"):
    length = Guardrail(name="short-enough", fn=lambda text: len(text) < 100)
    assert length.execute("ok").passed

# PDF: the core install ships no engine, so the two engine-free routes must
# work and the third must fail with an error that names the way out.
print("\nPDF without a local engine")

_MINIMAL_PDF = (
    b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
    b"trailer\n<< /Size 3 /Root 1 0 R >>\n%%EOF\n"
)

with check("PDF constructs and round-trips with no engine (native mode path)"):
    from fastaiagent import PDF

    pdf = PDF.from_bytes(_MINIMAL_PDF)
    assert pdf.media_type == "application/pdf"
    assert pdf.to_base64()
    assert PDF.from_dict(pdf.to_dict()).data == pdf.data

with check("bring-your-own text works with no engine"):
    byo = PDF.from_bytes(_MINIMAL_PDF, text="text from the caller's own parser")
    assert byo.extract_text() == "text from the caller's own parser"

with check("extract_text without an engine raises, naming every way out"):
    try:
        PDF.from_bytes(_MINIMAL_PDF).extract_text()
    except ImportError as e:  # MissingPDFBackend is also an ImportError, on purpose
        msg = str(e)
        assert "fastaiagent[pdf]" in msg, "error must name the extra"
        assert "text=" in msg, "error must name the bring-your-own escape hatch"
        assert 'pdf_mode="native"' in msg, "error must name the no-engine-needed mode"
    else:
        raise AssertionError("expected extract_text() to raise without an engine")

# The point of the exercise: none of the above may need a heavyweight or
# copyleft dependency. ``pymupdf`` is the AGPL one this gate exists for;
# the others are weight-only and merely confirm the core tree is slim.
print("\nabsent-by-design dependencies")
for module, why in (
    ("pymupdf", "AGPL-3.0 — must never be in the default install"),
    ("fitz", "AGPL-3.0 (pymupdf's legacy import name)"),
    ("pypdfium2", "permissive, but PDF decoding is the `pdf` extra's job"),
):
    with check(f"{module} is NOT installed ({why})"):
        assert importlib.util.find_spec(module) is None, (
            f"{module} is present in a core-only install — it must be confined to an extra"
        )

if failures:
    sys.stdout.flush()  # keep the per-check log above the summary in CI output
    print(f"\nFAIL: {len(failures)} core-surface check(s) failed", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    raise SystemExit(1)

print("\nOK: the core install carries the full library-usage surface.")
