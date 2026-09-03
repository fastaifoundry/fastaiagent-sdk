#!/usr/bin/env python3
"""Fail if any copyleft (AGPL/GPL) package is present in the installed tree.

Run this inside a venv built with ``pip install .`` — no extras, no ``[dev]`` —
to assert that a plain ``pip install fastaiagent`` stays permissively licensed.
Enterprise licence scanners flag AGPL anywhere in a resolved dependency tree and
procurement blocks on the flag, so this is the regression that must not return
silently when someone adds a convenient dependency.

    python scripts/check_licences.py            # fail on AGPL and GPL
    python scripts/check_licences.py --list     # print every distribution + licence

WHY THE OBVIOUS ONE-LINER DOESN'T WORK
A naive ``'AGPL' in dist.metadata['License']`` misses the exact package this
gate exists for. PyMuPDF declares::

    License: Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License

which contains "AFFERO GPL", not "AGPL" — so the substring check returns clean
while AGPL sits in the tree. Two more traps:

* Many distributions leave ``License`` empty and declare the licence *only* via
  ``Classifier: License :: OSI Approved :: ...`` (pypdfium2, Pillow, typer and
  regex all do this), or via the PEP 639 ``License-Expression`` field.
* "GPL" is a substring of "LGPL", which is not copyleft in the sense we care
  about here — a lookbehind is required, not a plain ``in``.

So: match ``AGPL|AFFERO|GPL`` (excluding ``LGPL``) across all three fields.
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib.metadata import distributions

# Matches AGPL / "Affero" / GPL, but not LGPL. ``(?<![A-Z])`` keeps "LGPL" and
# "SUPERGPL"-style false positives out while still matching a leading "GPL".
_COPYLEFT = re.compile(r"AFFERO|(?<![A-Z])A?GPL")
_AFFERO = re.compile(r"AFFERO|(?<![A-Z])AGPL")

# Distributions allowed to match despite the pattern, each with the reason.
# Intentionally empty: the core tree is clean today, and anything landing here
# must be a deliberate, reviewed decision rather than a quiet append.
ALLOWLIST: dict[str, str] = {}


def _licence_fields(dist: object) -> tuple[str, str, str]:
    """Return (License, License-Expression, joined License classifiers)."""
    meta = dist.metadata  # type: ignore[attr-defined]
    legacy = str(meta.get("License") or "")
    expression = str(meta.get("License-Expression") or "")
    classifiers = " ; ".join(
        c for c in (meta.get_all("Classifier") or []) if c.startswith("License")
    )
    return legacy, expression, classifiers


def scan() -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Return (affero_hits, gpl_hits) as (name, version, evidence) triples."""
    affero: list[tuple[str, str, str]] = []
    gpl: list[tuple[str, str, str]] = []
    for dist in distributions():
        name = str(dist.metadata.get("Name") or "<unknown>")
        if name in ALLOWLIST:
            continue
        legacy, expression, classifiers = _licence_fields(dist)
        blob = f"{legacy} {expression} {classifiers}".upper()
        if not _COPYLEFT.search(blob):
            continue
        evidence = "; ".join(p for p in (legacy, expression, classifiers) if p)
        entry = (name, dist.version, evidence)
        (affero if _AFFERO.search(blob) else gpl).append(entry)
    return affero, gpl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="print every distribution and its licence, then exit"
    )
    args = parser.parse_args()

    if args.list:
        for dist in sorted(distributions(), key=lambda d: str(d.metadata.get("Name") or "")):
            legacy, expression, classifiers = _licence_fields(dist)
            evidence = "; ".join(p for p in (legacy, expression, classifiers) if p) or "<none>"
            print(f"{str(dist.metadata.get('Name')):32} {dist.version:14} {evidence}")
        return 0

    affero, gpl = scan()
    total = len(list(distributions()))
    sys.stdout.flush()  # keep stdout/stderr ordered in CI output

    for name, version, evidence in affero:
        print(f"AGPL/Affero: {name} {version} -- {evidence}", file=sys.stderr)
    for name, version, evidence in gpl:
        print(f"GPL:         {name} {version} -- {evidence}", file=sys.stderr)

    if affero or gpl:
        print(
            f"\nFAIL: {len(affero) + len(gpl)} copyleft package(s) in the installed tree.\n"
            "A plain `pip install fastaiagent` must stay permissively licensed "
            "(MIT/BSD/Apache-2.0).\n"
            "Move the offending dependency to an opt-in extra, or replace it. If it is "
            "genuinely acceptable, add it to ALLOWLIST in this file with a written reason.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: no AGPL/GPL packages among {total} installed distributions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
