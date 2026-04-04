"""Evaluation results with summary and export."""

from __future__ import annotations

import json
from pathlib import Path

from fastaiagent.eval.scorer import ScorerResult


class EvalResults:
    """Results of an evaluation run."""

    def __init__(self, scores: dict[str, list[ScorerResult]] | None = None):
        self.scores: dict[str, list[ScorerResult]] = scores or {}

    def add(self, scorer_name: str, result: ScorerResult) -> None:
        self.scores.setdefault(scorer_name, []).append(result)

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
