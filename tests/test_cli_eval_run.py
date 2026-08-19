"""``fastaiagent eval run`` / ``eval compare`` — real CLI over real SQLite.

Exit-code contract under test: 0 gate passed · 1 quality failed · 3 infra
invalid (2 is reserved by Click for usage errors).
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from fastaiagent.cli.main import app

runner = CliRunner()

_AGENT_MODULE = """
def good(x):
    return {"greet": "hi", "farewell": "bye"}[x]

def regressed(x):
    return {"greet": "hi", "farewell": "WRONG"}[x]

def broken(x):
    raise TimeoutError("provider 500")

class RunStyleAgent:
    def run(self, x):
        return good(x)

agent_obj = RunStyleAgent()
"""

_DATASET = (
    '{"input": "greet", "expected_output": "hi"}\n{"input": "farewell", "expected_output": "bye"}\n'
)


def _setup(tmp_path):
    (tmp_path / "target.py").write_text(_AGENT_MODULE)
    (tmp_path / "cases.jsonl").write_text(_DATASET)
    return tmp_path / "target.py", tmp_path / "cases.jsonl", tmp_path / "local.db"


def test_run_gate_passes_exit_0(tmp_path) -> None:
    target, cases, db = _setup(tmp_path)
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--agent",
            f"{target}:good",
            "--dataset",
            str(cases),
            "--fail-under",
            "overall.pass_rate=0.9",
            "--db",
            str(db),
            "--run-name",
            "main",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "gate PASSED" in result.output


def test_run_quality_fail_exit_1(tmp_path) -> None:
    target, cases, db = _setup(tmp_path)
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--agent",
            f"{target}:regressed",
            "--dataset",
            str(cases),
            "--fail-under",
            "overall.pass_rate=0.9",
            "--db",
            str(db),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "FAILED" in result.output


def test_run_infra_invalid_exit_3(tmp_path) -> None:
    target, cases, db = _setup(tmp_path)
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--agent",
            f"{target}:broken",
            "--dataset",
            str(cases),
            "--fail-under",
            "overall.pass_rate=0.9",
            "--max-error-rate",
            "0.1",
            "--db",
            str(db),
        ],
    )
    assert result.exit_code == 3, result.output
    assert "INVALID" in result.output


def test_run_accepts_agent_object_with_run_method(tmp_path) -> None:
    target, cases, db = _setup(tmp_path)
    result = runner.invoke(
        app,
        ["eval", "run", "--agent", f"{target}:agent_obj", "--dataset", str(cases), "--db", str(db)],
    )
    assert result.exit_code == 0, result.output


def test_run_writes_json_report(tmp_path) -> None:
    target, cases, db = _setup(tmp_path)
    report_path = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--agent",
            f"{target}:good",
            "--dataset",
            str(cases),
            "--fail-under",
            "overall.pass_rate=0.9",
            "--db",
            str(db),
            "--json",
            str(report_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(report_path.read_text())
    assert payload["schema_version"] == 1
    assert payload["gate"]["outcome"] == "passed"
    assert payload["scorecard"]["overall_pass_rate"] == 1.0
    assert payload["run_id"]


def test_run_bad_target_is_usage_error(tmp_path) -> None:
    _, cases, db = _setup(tmp_path)
    result = runner.invoke(
        app,
        ["eval", "run", "--agent", "no-colon-here", "--dataset", str(cases), "--db", str(db)],
    )
    assert result.exit_code == 2  # Click usage error


def test_compare_regression_exit_1_and_tolerance(tmp_path) -> None:
    target, cases, db = _setup(tmp_path)
    for name, fn in (("main", "good"), ("pr", "regressed")):
        r = runner.invoke(
            app,
            [
                "eval",
                "run",
                "--agent",
                f"{target}:{fn}",
                "--dataset",
                str(cases),
                "--db",
                str(db),
                "--run-name",
                name,
            ],
        )
        assert r.exit_code in (0, 1)

    result = runner.invoke(app, ["eval", "compare", "main", "pr", "--db", str(db)])
    assert result.exit_code == 1, result.output
    assert "REGRESSION" in result.output

    result = runner.invoke(
        app, ["eval", "compare", "main", "pr", "--db", str(db), "--tolerance", "0.6"]
    )
    assert result.exit_code == 0, result.output


def test_compare_missing_run_exit_3(tmp_path) -> None:
    target, cases, db = _setup(tmp_path)
    runner.invoke(
        app,
        [
            "eval",
            "run",
            "--agent",
            f"{target}:good",
            "--dataset",
            str(cases),
            "--db",
            str(db),
            "--run-name",
            "main",
        ],
    )
    result = runner.invoke(app, ["eval", "compare", "main", "ghost", "--db", str(db)])
    assert result.exit_code == 3, result.output
    assert "not found" in result.output
