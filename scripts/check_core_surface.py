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

# The point of the exercise: none of the above may need a heavyweight or
# copyleft dependency. ``pymupdf`` is the AGPL one this gate exists for;
# the others are weight-only and merely confirm the core tree is slim.
print("\nabsent-by-design dependencies")
for module, why in (
    ("pymupdf", "AGPL-3.0 — must never be in the default install"),
    ("fitz", "AGPL-3.0 (pymupdf's legacy import name)"),
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
