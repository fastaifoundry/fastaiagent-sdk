"""Smoke tests for the AutoLLM loop — no live LLM calls.

Run from this folder (CI does): ``pytest tests/``. Covers what every step relies
on before anyone spends tokens: the scripts import, the data encodes each house
rule often enough to be learned, the scorer grades as documented, and
``load_agent`` rebuilds a registry version the way ``report.apply_to()`` would.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import triage  # noqa: E402

from fastaiagent.testing import FunctionModel  # noqa: E402

STEPS = sorted(p.name for p in _HERE.glob("0*.py"))


@pytest.mark.parametrize("script", STEPS)
def test_every_step_imports_without_running(script: str) -> None:
    spec = importlib.util.spec_from_file_location(script[:-3], _HERE / script)
    assert spec and spec.loader
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def test_the_tickets_are_well_formed() -> None:
    rows = triage.tickets()
    assert len(rows) == 120
    assert len({r["id"] for r in rows}) == len({r["ticket"] for r in rows}) == 120
    for r in rows:
        assert r["label"]["queue"] in {"billing", "technical", "account", "shipping", "security"}
        assert r["label"]["priority"] in {"P1", "P2", "P3"}


def test_every_house_rule_has_enough_cases_to_learn() -> None:
    """Each outcome of each rule needs ~10 cases, so every split sees some."""
    counts = Counter((r["label"]["queue"], r["label"]["priority"]) for r in triage.tickets())
    for outcome in [
        ("billing", "P1"),  # dispute over EUR 500
        ("billing", "P3"),  # EUR 500 or less
        ("security", "P1"),
        ("technical", "P1"),  # outage on Enterprise
        ("technical", "P2"),  # outage on Free/Team
        ("technical", "P3"),  # how-to
        ("shipping", "P2"),  # 5+ business days late
        ("shipping", "P3"),  # less
    ]:
        assert counts[outcome] >= 10, outcome
    assert counts[("account", "P2")] >= 20  # lockouts + GDPR requests


def test_v1_does_not_already_know_the_house_rules() -> None:
    """The point of the loop: v1 names queues and format, not the thresholds."""
    for rule_hint in ("500", "Enterprise", "business day", "GDPR"):
        assert rule_hint not in triage.V1_PROMPT


@pytest.mark.parametrize(
    ("output", "score", "passed", "reason"),
    [
        ('{"queue": "billing", "priority": "P1"}', 1.0, True, None),
        ('{"queue": "billing", "priority": "P2"}', 0.5, False, "priority P2, expected P1"),
        ('{"queue": "account", "priority": "P3"}', 0.0, False, "queue account, expected billing"),
        ("billing, P1", 0.0, False, "not a JSON object: 'billing, P1'"),
    ],
)
def test_triage_match_grades_as_documented(
    output: str, score: float, passed: bool, reason: str | None
) -> None:
    expected = triage.label_text({"queue": "billing", "priority": "P1"})
    result = triage.TriageMatch().score("ticket", output, expected)
    assert (result.score, result.passed) == (score, passed)
    assert result.reason == reason or (reason and result.reason.startswith(reason))


@pytest.fixture()
def registry(isolated_db: Path) -> Any:
    from fastaiagent.prompt import PromptRegistry

    reg = PromptRegistry()
    reg.register(triage.PROMPT, triage.V1_PROMPT)
    demos = [{"input": "charged EUR 900 twice", "output": '{"queue": "billing", "priority": "P1"}'}]
    reg.register(triage.PROMPT, "v2 prompt", metadata={"fewshot_demos": demos})
    reg.set_alias(triage.PROMPT, 1, "production")
    return reg


@pytest.fixture()
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from fastaiagent._internal.config import reset_config

    db = tmp_path / ".fastaiagent" / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    reset_config()
    yield db
    reset_config()


def test_load_agent_builds_from_the_alias(registry: Any) -> None:
    agent = triage.load_agent(alias="production")
    assert agent.system_prompt == triage.V1_PROMPT
    assert agent.memory is None
    assert agent._prompt_provenance == {"name": triage.PROMPT, "version": 1}


def test_load_agent_rebuilds_the_whole_winner_and_stays_stateless(registry: Any) -> None:
    agent = triage.load_agent(version=2)
    assert agent.system_prompt == "v2 prompt"
    assert agent._prompt_provenance == {"name": triage.PROMPT, "version": 2}

    seen: list[list[str]] = []

    def model(messages: list[Any]) -> str:
        seen.append([str(m.content) for m in messages])
        return '{"queue": "billing", "priority": "P1"}'

    agent.llm = FunctionModel(model)
    agent.run("first customer: refund EUR 700")
    agent.run("second customer: refund EUR 20")

    second = seen[-1]
    assert any("charged EUR 900 twice" in m for m in second)  # the winner's example
    assert not any("first customer" in m for m in second)  # nothing carried over


def test_the_dataset_lands_where_the_ui_reads_it(isolated_db: Path) -> None:
    assert triage.dataset_path() == isolated_db.parent / "datasets" / "ticket-triage.jsonl"


def test_label_text_is_what_a_correct_agent_replies() -> None:
    assert json.loads(triage.label_text({"queue": "shipping", "priority": "P2"})) == {
        "queue": "shipping",
        "priority": "P2",
    }
