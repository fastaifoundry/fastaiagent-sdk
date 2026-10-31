"""CLI commands that print JSON print valid JSON when redirected (1.95.0, S4).

They printed through Rich, which wraps long strings at the terminal width (80
columns when stdout is not a TTY), so ``fastaiagent traces export <id> > trace.json``
wrote a file ``json.load`` rejected. ``CliRunner`` is not a TTY either.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from fastaiagent.cli.main import app
from fastaiagent.trace import otel, trace_context

runner = CliRunner()
LONG = "a long attribute value that is well past eighty columns wide " * 4


def test_traces_export_is_valid_json(isolated_local_db: Path) -> None:
    otel.reset()
    try:
        with trace_context("long-trace") as span:
            span.set_attribute("note", LONG)
            trace_id = format(span.get_span_context().trace_id, "032x")
        otel.get_tracer_provider().force_flush()
    finally:
        otel.reset()

    result = runner.invoke(app, ["traces", "export", trace_id])
    assert result.exit_code == 0, result.output
    exported = json.loads(result.stdout)
    assert exported["trace_id"] == trace_id
    assert exported["spans"][0]["attributes"]["note"] == LONG


def test_eval_export_dry_run_stdout_is_valid_json(isolated_local_db: Path) -> None:
    from fastaiagent.eval import Dataset, evaluate

    evaluate(
        agent_fn=lambda _: LONG,
        dataset=Dataset.from_list([{"input": "q", "expected": LONG}]),
        scorers=["exact_match"],
        run_name=LONG,
    )

    result = runner.invoke(app, ["eval", "export", "--dry-run", "--db", str(isolated_local_db)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert len(payload["runs"]) == 1
    assert "run(s)" in result.stderr  # the notes still reach the terminal
