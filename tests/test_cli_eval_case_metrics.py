"""``fastaiagent eval run`` / ``compare`` with 1.89.0's case metrics. Real CLI, real SQLite.

* ``--fail-under cases.pass_rate=…`` gates on cases; ``overall.avg_score`` is
  refused (exit 3) instead of silently gating on the pass rate.
* A dataset no scorer applies to is **invalid** (exit 3), not a pass.
* ``--json`` carries ``scorecard.cases`` and ``gate.invalid_reasons``.
* ``compare`` prints errored and unmatched rows.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from fastaiagent.cli.main import app

runner = CliRunner()

_AGENT_MODULE = """
def half(x):
    return {"a": "a", "b": "WRONG", "c": "c", "d": "d"}[x]

def boom_on_c(x):
    if x == "c":
        raise RuntimeError("BOOM")
    return x
"""

_DATASET = "".join(
    json.dumps({"input": k, "expected_output": k}) + "\n" for k in ("a", "b", "c", "d")
)
_NO_EXPECTED = "".join(json.dumps({"input": k}) + "\n" for k in ("a", "b"))


def _setup(tmp_path: Path, dataset: str = _DATASET) -> tuple[Path, Path, Path]:
    (tmp_path / "target.py").write_text(_AGENT_MODULE)
    (tmp_path / "cases.jsonl").write_text(dataset)
    return tmp_path / "target.py", tmp_path / "cases.jsonl", tmp_path / "local.db"


def _run(target: Path, cases: Path, db: Path, fn: str, *extra: str) -> object:
    return runner.invoke(
        app,
        ["eval", "run", "--agent", f"{target}:{fn}", "--dataset", str(cases), "--db", str(db)]
        + list(extra),
    )


def test_cases_pass_rate_gate_and_json(tmp_path: Path) -> None:
    target, cases, db = _setup(tmp_path)
    out = tmp_path / "report.json"
    result = _run(
        target, cases, db, "half", "--fail-under", "cases.pass_rate=0.8", "--json", str(out)
    )
    assert result.exit_code == 1, result.output
    report = json.loads(out.read_text())
    assert report["scorecard"]["cases"]["pass_rate"] == 0.75
    assert report["gate"]["invalid_reasons"] == []
    assert "cases" in result.output and "overall (checks)" in result.output


def test_overall_avg_score_is_refused(tmp_path: Path) -> None:
    target, cases, db = _setup(tmp_path)
    result = _run(target, cases, db, "half", "--fail-under", "overall.avg_score=0.5")
    assert result.exit_code == 3, result.output
    assert "overall.avg_score is not a metric" in result.output


def test_nothing_applies_is_invalid_not_a_pass(tmp_path: Path) -> None:
    target, cases, db = _setup(tmp_path, _NO_EXPECTED)
    result = _run(target, cases, db, "half", "--fail-under", "overall.pass_rate=0.5")
    assert result.exit_code == 3, result.output
    assert "INVALID" in result.output


def test_compare_prints_errored_and_unmatched(tmp_path: Path) -> None:
    target, cases, db = _setup(tmp_path)
    assert _run(target, cases, db, "half", "--run-name", "base").exit_code == 0
    assert _run(target, cases, db, "boom_on_c", "--run-name", "new").exit_code == 0
    result = runner.invoke(
        app, ["eval", "compare", "base", "new", "--db", str(db), "--no-fail-on-regression"]
    )
    assert result.exit_code == 0, result.output
    assert "errored (either run)" in result.output
    assert "only in baseline / current" in result.output
    assert "cases pass_rate delta" in result.output
