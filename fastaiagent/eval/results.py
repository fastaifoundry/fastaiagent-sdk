"""Evaluation results with summary, export, and local persistence."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastaiagent.eval.scorer import ScorerResult


@dataclass
class EvalCaseRecord:
    """Per-case capture used by :meth:`EvalResults.persist_local`."""

    input: Any = None
    expected_output: Any = None
    actual_output: Any = None
    trace_id: str | None = None
    per_scorer: dict[str, dict[str, Any]] = field(default_factory=dict)


class EvalResults:
    """Results of an evaluation run."""

    def __init__(self, scores: dict[str, list[ScorerResult]] | None = None):
        self.scores: dict[str, list[ScorerResult]] = scores or {}
        self.cases: list[EvalCaseRecord] = []

    def add(self, scorer_name: str, result: ScorerResult) -> None:
        self.scores.setdefault(scorer_name, []).append(result)

    def add_case(self, record: EvalCaseRecord) -> None:
        """Record one dataset case end-to-end for later persistence."""
        self.cases.append(record)

    def summary(self) -> str:
        """Generate a summary table."""
        lines = ["Evaluation Results", "=" * 50]
        for name, results in self.scores.items():
            if not results:
                continue
            avg_score = sum(r.score for r in results) / len(results)
            pass_rate = sum(1 for r in results if r.passed) / len(results)
            lines.append(
                f"{name}: avg={avg_score:.2f} pass_rate={pass_rate:.0%} ({len(results)} cases)"
            )
        return "\n".join(lines)

    def export(self, path: str | Path, format: str = "json") -> None:
        """Export results to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: [r.model_dump() for r in results] for name, results in self.scores.items()}
        path.write_text(json.dumps(data, indent=2))

    def publish(self, run_name: str | None = None) -> None:
        """Publish eval results to platform."""
        from fastaiagent._internal.errors import PlatformNotConnectedError
        from fastaiagent._platform.api import get_platform_api
        from fastaiagent.client import _connection

        if not _connection.is_connected:
            raise PlatformNotConnectedError(
                "Not connected to platform. Call fa.connect() first."
            )
        api = get_platform_api()
        data = {name: [r.model_dump() for r in results] for name, results in self.scores.items()}
        api.post(
            "/public/v1/eval/runs",
            {"run_name": run_name, "scores": data},
        )

    def persist_local(
        self,
        *,
        db_path: str | Path | None = None,
        run_name: str | None = None,
        dataset_name: str | None = None,
        agent_name: str | None = None,
        agent_version: str | None = None,
    ) -> str:
        """Persist this run to the unified local.db.

        Writes one row to ``eval_runs`` and one per case to ``eval_cases``.
        Returns the generated ``run_id`` so callers can correlate.
        """
        from fastaiagent._internal.config import get_config
        from fastaiagent.ui.db import init_local_db

        resolved = Path(db_path) if db_path is not None else Path(get_config().local_db_path)
        run_id = uuid.uuid4().hex
        timestamp = datetime.now(tz=timezone.utc).isoformat()

        pass_count = 0
        fail_count = 0
        total = 0
        for results in self.scores.values():
            for r in results:
                total += 1
                if r.passed:
                    pass_count += 1
                else:
                    fail_count += 1
        pass_rate = (pass_count / total) if total else 0.0

        db = init_local_db(resolved)
        try:
            db.execute(
                """INSERT INTO eval_runs
                   (run_id, run_name, dataset_name, agent_name, agent_version,
                    scorers, started_at, finished_at, pass_count, fail_count,
                    pass_rate, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    run_name,
                    dataset_name,
                    agent_name,
                    agent_version,
                    json.dumps(sorted(self.scores.keys())),
                    timestamp,
                    timestamp,
                    pass_count,
                    fail_count,
                    pass_rate,
                    json.dumps({}),
                ),
            )
            for ordinal, case in enumerate(self.cases):
                db.execute(
                    """INSERT INTO eval_cases
                       (case_id, run_id, ordinal, input, expected_output,
                        actual_output, trace_id, per_scorer)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        uuid.uuid4().hex,
                        run_id,
                        ordinal,
                        json.dumps(case.input, default=str),
                        json.dumps(case.expected_output, default=str),
                        json.dumps(case.actual_output, default=str),
                        case.trace_id,
                        json.dumps(case.per_scorer),
                    ),
                )
        finally:
            db.close()
        return run_id

    def compare(self, other: EvalResults) -> str:
        """Compare with another set of results."""
        lines = ["Comparison", "=" * 50]
        all_scorers = set(self.scores.keys()) | set(other.scores.keys())
        for name in sorted(all_scorers):
            a = self.scores.get(name, [])
            b = other.scores.get(name, [])
            avg_a = sum(r.score for r in a) / max(len(a), 1)
            avg_b = sum(r.score for r in b) / max(len(b), 1)
            diff = avg_b - avg_a
            sign = "+" if diff > 0 else ""
            lines.append(f"{name}: {avg_a:.2f} → {avg_b:.2f} ({sign}{diff:.2f})")
        return "\n".join(lines)
