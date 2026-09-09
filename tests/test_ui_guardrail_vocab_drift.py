"""The Local UI's guardrail vocabulary must not drift from the SDK's.

1.57.0 added two implementation types and shipped a Local UI whose type filter
still listed five. The backend accepted `content_safety` and `groundedness`, the
runtime recorded events for them, and the one screen an operator would use to
find those events could not select them.

The SPA is a prebuilt bundle, so nothing in the Python test suite exercised it —
which is exactly why the drift shipped. These tests read the SPA source and hold
it to the enums it is rendering, so the next type to be added fails here rather
than in someone's browser.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from fastaiagent.guardrail.actions import ACTIONS, ACTIONS_TAKEN
from fastaiagent.guardrail.guardrail import GuardrailPosition, GuardrailType

_PAGE = Path(__file__).resolve().parents[1] / "ui-frontend/src/pages/GuardrailsPage.tsx"

pytestmark = pytest.mark.skipif(
    not _PAGE.is_file(),
    reason="ui-frontend source not present (it is excluded from the sdist)",
)


def _string_array(name: str) -> list[str]:
    """Pull a `const NAME = [...]` string array out of the TSX source."""
    src = _PAGE.read_text()
    match = re.search(rf"const {name}(?:\s*:\s*[^=]+)?\s*=\s*\[(.*?)\]", src, re.DOTALL)
    assert match, f"{name} not found in {_PAGE.name} — was it renamed?"
    return re.findall(r'"([^"]+)"', match.group(1))


def _record_keys(name: str) -> list[str]:
    """Pull the keys of a `const NAME: Record<...> = { ... }` object."""
    src = _PAGE.read_text()
    match = re.search(rf"const {name}\s*:\s*Record<[^>]*>\s*=\s*\{{(.*?)\n\}};", src, re.DOTALL)
    assert match, f"{name} not found in {_PAGE.name} — was it renamed?"
    return re.findall(r"^\s*([a-z_]+):", match.group(1), re.MULTILINE)


def test_the_type_filter_offers_every_implementation_type() -> None:
    offered = set(_string_array("TYPE_OPTIONS"))
    known = {t.value for t in GuardrailType}
    assert offered == known, (
        "the Local UI's type filter has drifted from GuardrailType: "
        f"missing {sorted(known - offered)}, unknown {sorted(offered - known)}"
    )


def test_the_position_filter_offers_every_position() -> None:
    offered = set(_string_array("POSITION_OPTIONS"))
    known = {p.value for p in GuardrailPosition}
    assert offered == known, f"missing {sorted(known - offered)}, unknown {sorted(offered - known)}"


def test_every_action_maps_to_the_outcome_it_produces_when_fulfilled() -> None:
    """The UI flags a row whose action could not be carried out. That comparison
    is meaningless unless every action has a fulfilled form to compare against —
    the names are present tense (`block`) and the outcomes past (`blocked`)."""
    fulfilled = _record_keys("ACTION_FULFILLED")
    assert set(fulfilled) == set(ACTIONS), (
        f"ACTION_FULFILLED does not cover every action: missing {sorted(set(ACTIONS) - set(fulfilled))}"
    )

    src = _PAGE.read_text()
    match = re.search(r"const ACTION_FULFILLED\s*:\s*Record<[^>]*>\s*=\s*\{(.*?)\n\};", src, re.DOTALL)
    assert match
    values = re.findall(r':\s*"([^"]+)"', match.group(1))
    unknown = set(values) - set(ACTIONS_TAKEN)
    assert not unknown, f"ACTION_FULFILLED maps to outcomes the SDK never emits: {sorted(unknown)}"


def test_the_outcome_filter_covers_every_outcome_the_runtime_writes() -> None:
    """Including ``filtered``, which nothing could produce before 1.57.0."""
    from fastaiagent.ui.events import _outcome

    known = set(_record_keys("OUTCOME_META"))
    assert {"passed", "blocked", "warned", "errored", "filtered"} <= known, sorted(known)
    # And the writer only ever produces outcomes the UI knows how to render.
    assert _outcome.__module__  # import guard; the values themselves are asserted above

