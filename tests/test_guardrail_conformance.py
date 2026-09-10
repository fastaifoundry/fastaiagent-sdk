"""Run the shared guardrail conformance fixture against the SDK's implementation.

``tests/data/guardrail_conformance.json`` exists in this repo and in the plane's
(`backend/app/data/guardrail_conformance.json`), **byte-identical**, and both
sides run it through a thin adapter. That is the point: agreement between the two
repos stops being a matter of each session reading the other's prose and becomes
a test run.

**Why it exists.** `pii`/`secrets` were built on both sides from a written
handover, faithfully, and still diverged four ways. A type's contract spans TWO
modules per side — the detector and the config resolver — and only the detector
was pinned by a mirror test. `{"entities": []}` raised on the plane and **passed**
here: a detection control reporting success while inspecting nothing, the exact
defect the `schema` type carried until 1.59.0.

Ownership, from `docs/Guardrail_Type_Contract.md` §1: the **detector** is ours and
the plane mirrors it; the **config resolver** and the **result shape** are the
plane's and we mirror them. Different halves, different directions — which is
what made the drift invisible to a mirror scoped to one file.

**The protocol** (§2): add cases before changing behaviour, never edit one copy,
and on a mismatch decide which side owns it and change both. In particular —
``expect.raises`` means the check **could not run**, never "found nothing".

Hermetic: no network, no plane, no model. `backend/tests/test_detectors_mirror.py`
on their side byte-compares the two copies and starts enforcing once this file
exists.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from typing import Any

import pytest

from fastaiagent.guardrail.actions import mask_payload
from fastaiagent.guardrail.guardrail import Guardrail, GuardrailType
from fastaiagent.guardrail.implementations import _run_pii, _run_secrets

FIXTURE = pathlib.Path(__file__).resolve().parent / "data" / "guardrail_conformance.json"
_CASES = json.loads(FIXTURE.read_text())


def _rule(impl: str, config: dict[str, Any]) -> Guardrail:
    return Guardrail(name=f"conformance-{impl}", guardrail_type=GuardrailType(impl), config=config)


def _detail(impl: str, config: dict[str, Any], text: str) -> dict[str, Any]:
    """The SDK's adapter: resolve → detect → summarise.

    Deliberately the runner itself rather than a reimplementation of it, so a
    case that passes here is a case the runtime handles. Any of the three steps
    may raise; the caller decides what that means. Note we call the runner
    *directly*, not through ``run_guardrail`` — that wraps exceptions into an
    ``errored`` result per ``on_error``, and here we want the raise.
    """
    runner = _run_pii if impl == "pii" else _run_secrets
    return asyncio.run(runner(_rule(impl, config), text)).metadata


def _masked(impl: str, config: dict[str, Any], text: str) -> str:
    out = asyncio.run(mask_payload(_rule(impl, config), text))
    # ``mask_payload`` answers "did anything change?", so ``None`` is the
    # unchanged payload — which is what the caller carries forward.
    return text if out is None else out


def _ids(cases: list[dict[str, Any]]) -> list[str]:
    return [c["name"] for c in cases]


def _run_detection_case(impl: str, case: dict[str, Any]) -> None:
    expect = case["expect"]
    if expect.get("raises"):
        # `raises` means the check could not run. It must never resolve to "found
        # nothing" — that reads as a healthy control over an uninspected payload.
        with pytest.raises((ValueError, ImportError)):
            _detail(impl, case["config"], case["input"])
        return
    assert _detail(impl, case["config"], case["input"]) == expect["detail"]


@pytest.mark.parametrize("case", _CASES["pii"], ids=_ids(_CASES["pii"]))
def test_pii_conformance(case: dict[str, Any]) -> None:
    _run_detection_case("pii", case)


@pytest.mark.parametrize("case", _CASES["secrets"], ids=_ids(_CASES["secrets"]))
def test_secrets_conformance(case: dict[str, Any]) -> None:
    _run_detection_case("secrets", case)


@pytest.mark.parametrize("case", _CASES["mask"], ids=_ids(_CASES["mask"]))
def test_mask_conformance(case: dict[str, Any]) -> None:
    assert _masked(case["type"], case["config"], case["input"]) == case["expect"]["masked"]


def test_the_fixture_covers_both_types_and_masking() -> None:
    """A fixture that quietly lost a section would pass every test above while
    checking less."""
    assert {"pii", "secrets", "mask"} <= set(_CASES)
    for key in ("pii", "secrets", "mask"):
        assert _CASES[key], f"{key} section is empty"
    # The case that started this. If someone removes it, the divergence it pins
    # can come back.
    assert any(c["config"].get("entities") == [] for c in _CASES["pii"]), (
        "the empty-entity-list case is load-bearing: it is where the two sides disagreed"
    )
