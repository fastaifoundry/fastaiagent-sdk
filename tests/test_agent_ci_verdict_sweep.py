"""The verdict an Agent-CI session records is the verdict it enforces (1.89.0).

E1. The pytest plugin computed two independent verdicts: the threshold gate's
``report.outcome`` and the baseline check. It OR-ed both into the exit status
but recorded only the first, so a session that exited 1 on a regression, or
on a missing baseline, told the plane ``gate_outcome="passed"``. The plane
counts only ``passed`` as promotion and compliance evidence.

E2. ``--eval-run-name main --eval-baseline main`` persisted the session first
and then looked ``main`` up by name, so it compared the session with itself:
delta 0, and the regressed run became the next baseline.

This sweep runs real inner pytest sessions (pytester, no mocks) over every
gate configuration and asserts the one invariant that would have caught
both: **the recorded verdict is ``passed`` exactly when the session passed**,
and the payload the plane would receive carries that same verdict.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any

import pytest

pytest_plugins = ["pytester"]

#: The plane's ingest Literal. Inlined, not imported, so this file fails on
#: 1.88.0 by assertion rather than ImportError.
VALID = ("passed", "failed", "invalid")


@pytest.fixture(autouse=True)
def _isolate_config():
    from fastaiagent._internal.config import reset_config

    yield
    os.environ.pop("FASTAIAGENT_LOCAL_DB", None)
    reset_config()


_HEADER = """
import os
os.environ["FASTAIAGENT_LOCAL_DB"] = r"{db}"
from fastaiagent._internal.config import reset_config
reset_config()

from fastaiagent.testing import TestModel
from fastaiagent.agent import Agent
from fastaiagent.eval import case
"""

SUITES = {
    "passing": _HEADER
    + """
@case(input="hello", expected="hi")
def test_a(evaluate_one):
    evaluate_one(Agent(name="g", llm=TestModel(response="hi")).run, scorers=["exact_match"])

@case(input="bye", expected="bye")
def test_b(evaluate_one):
    evaluate_one(Agent(name="g", llm=TestModel(response="bye")).run, scorers=["exact_match"])
""",
    "mixed": _HEADER
    + """
@case(input="hello", expected="hi")
def test_a(evaluate_one):
    evaluate_one(Agent(name="g", llm=TestModel(response="hi")).run, scorers=["exact_match"])

@case(input="bye", expected="bye")
def test_b(evaluate_one):
    agent = Agent(name="g", llm=TestModel(response="WRONG"))
    evaluate_one(agent.run, scorers=["exact_match"], assert_pass=False)
""",
    "infra": _HEADER
    + """
def _broken(x):
    raise TimeoutError("provider 500")

@case(input="hello", expected="hi")
def test_a(evaluate_one):
    evaluate_one(_broken, scorers=["exact_match"])
""",
}


@dataclass
class Session:
    suite: str
    args: list[str] = field(default_factory=list)


@dataclass
class Config:
    setup: list[Session]
    final: Session
    outcome: str
    #: The session whose run the final one must have been compared against
    #: (index into ``setup``), or None.
    baseline_from: int | None = None


CONFIGS = {
    "no_gate": Config([], Session("mixed"), "passed"),
    "fail_under_met": Config(
        [], Session("passing", ["--eval-fail-under", "overall.pass_rate=0.9"]), "passed"
    ),
    "fail_under_missed": Config(
        [], Session("mixed", ["--eval-fail-under", "overall.pass_rate=0.9"]), "failed"
    ),
    "max_error_rate_exceeded": Config(
        [], Session("infra", ["--eval-max-error-rate", "0.1"]), "invalid"
    ),
    "baseline_held": Config(
        [Session("passing", ["--eval-run-name", "main"])],
        Session("passing", ["--eval-baseline", "main"]),
        "passed",
        baseline_from=0,
    ),
    "baseline_regressed": Config(
        [Session("passing", ["--eval-run-name", "main"])],
        Session("mixed", ["--eval-baseline", "main"]),
        "failed",
        baseline_from=0,
    ),
    "baseline_missing": Config(
        [], Session("passing", ["--eval-baseline", "does-not-exist"]), "invalid"
    ),
    "thresholds_met_but_regressed": Config(
        [Session("passing", ["--eval-run-name", "main"])],
        Session("mixed", ["--eval-fail-under", "overall.pass_rate=0.4", "--eval-baseline", "main"]),
        "failed",
        baseline_from=0,
    ),
    "regressed_within_tolerance": Config(
        [Session("passing", ["--eval-run-name", "main"])],
        Session("mixed", ["--eval-baseline", "main", "--eval-tolerance", "0.6"]),
        "passed",
        baseline_from=0,
    ),
    # E2: the ratchet — publish and compare under one name.
    "same_name_compares_previous_run": Config(
        [Session("passing", ["--eval-run-name", "main"])],
        Session("mixed", ["--eval-run-name", "main", "--eval-baseline", "main"]),
        "failed",
        baseline_from=0,
    ),
    # D1b: the first run of a ratchet has nothing to compare with yet.
    "same_name_bootstrap": Config(
        [], Session("passing", ["--eval-run-name", "main", "--eval-baseline", "main"]), "passed"
    ),
    # D3: a run that failed its gate never becomes the next baseline.
    "failed_run_is_not_the_next_baseline": Config(
        [
            Session("passing", ["--eval-run-name", "main"]),
            Session("mixed", ["--eval-run-name", "main", "--eval-baseline", "main"]),
        ],
        Session("mixed", ["--eval-run-name", "main", "--eval-baseline", "main"]),
        "failed",
        baseline_from=0,
    ),
}


def _run(pytester: pytest.Pytester, db: Any, session: Session) -> int:
    pytester.makepyfile(SUITES[session.suite].format(db=db))
    return int(pytester.runpytest("--no-header", "-p", "no:randomly", *session.args).ret)


def _latest_run(db: Any) -> tuple[str, dict[str, Any]]:
    with sqlite3.connect(db) as conn:
        run_id, metadata = conn.execute(
            "SELECT run_id, metadata FROM eval_runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    return run_id, json.loads(metadata or "{}")


@pytest.mark.parametrize("name", list(CONFIGS))
def test_the_recorded_verdict_is_the_enforced_one(
    pytester: pytest.Pytester, tmp_path: Any, name: str
) -> None:
    from fastaiagent.eval.platform_export import build_payloads

    cfg = CONFIGS[name]
    db = tmp_path / "local.db"
    setup_ids = []
    for s in cfg.setup:
        _run(pytester, db, s)
        setup_ids.append(_latest_run(db)[0])

    ret = _run(pytester, db, cfg.final)
    run_id, meta = _latest_run(db)
    recorded = meta.get("gate_outcome")

    assert recorded in VALID, f"{name}: recorded {recorded!r}"
    assert (recorded == "passed") == (ret == 0), (
        f"{name}: the session exited {ret} but recorded gate_outcome={recorded!r}"
    )
    assert recorded == cfg.outcome, f"{name}: expected {cfg.outcome!r}, recorded {recorded!r}"

    payload = build_payloads(run_id=run_id, db_path=db)[0]
    assert payload["gate_outcome"] == recorded, "the plane would receive a different verdict"

    if cfg.baseline_from is not None:
        baseline = meta.get("baseline") or {}
        assert baseline.get("run_id") == setup_ids[cfg.baseline_from], (
            f"{name}: compared against {baseline.get('run_id')!r}, "
            f"expected the earlier run {setup_ids[cfg.baseline_from]!r}"
        )
        assert baseline.get("run_id") != run_id, "the session was compared with itself"


def test_a_malformed_threshold_is_a_usage_error_and_persists_nothing(
    pytester: pytest.Pytester, tmp_path: Any
) -> None:
    """It used to raise inside sessionfinish, after persisting a verdict-less run."""
    db = tmp_path / "local.db"
    pytester.makepyfile(SUITES["passing"].format(db=db))
    result = pytester.runpytest("--no-header", "--eval-fail-under", "overall.pass_rate=lots")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert _run_count(db) == 0, "a run was persisted for a session that never ran"


def _run_count(db: Any) -> int:
    if not db.exists():
        return 0
    with sqlite3.connect(db) as conn:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='eval_runs'"
        ).fetchone()
        return conn.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0] if has_table else 0


# ---------------------------------------------------------------------------
# The CLI axis: exit code and recorded verdict agree
# ---------------------------------------------------------------------------

_AGENT_MODULE = """
def good(x):
    return {"greet": "hi", "farewell": "bye"}[x]

def regressed(x):
    return {"greet": "hi", "farewell": "WRONG"}[x]

def broken(x):
    raise TimeoutError("provider 500")
"""

_DATASET = (
    '{"input": "greet", "expected_output": "hi"}\n{"input": "farewell", "expected_output": "bye"}\n'
)


@pytest.mark.parametrize(
    ("agent", "extra", "exit_code", "outcome"),
    [
        ("good", [], 0, "passed"),
        ("regressed", [], 1, "failed"),
        ("broken", ["--max-error-rate", "0.1"], 3, "invalid"),
    ],
)
def test_cli_exit_code_matches_recorded_verdict(
    tmp_path: Any, agent: str, extra: list[str], exit_code: int, outcome: str
) -> None:
    from typer.testing import CliRunner

    from fastaiagent.cli.main import app

    (tmp_path / "target.py").write_text(_AGENT_MODULE)
    (tmp_path / "cases.jsonl").write_text(_DATASET)
    db = tmp_path / "local.db"
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--agent",
            f"{tmp_path / 'target.py'}:{agent}",
            "--dataset",
            str(tmp_path / "cases.jsonl"),
            "--fail-under",
            "overall.pass_rate=0.9",
            "--db",
            str(db),
            *extra,
        ],
    )
    assert result.exit_code == exit_code, result.output
    assert _latest_run(db)[1].get("gate_outcome") == outcome
