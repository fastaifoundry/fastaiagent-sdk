"""Tests for fastaiagent.eval module."""

from __future__ import annotations

import json

import pytest

from fastaiagent.eval import Dataset, EvalResults, Scorer, ScorerResult, evaluate
from fastaiagent.eval.builtins import Contains, ExactMatch, JSONValid, LengthBetween
from fastaiagent.eval.trajectory import (
    CycleEfficiency,
    PathCorrectness,
    StepEfficiency,
    ToolUsageAccuracy,
)


class TestDataset:
    def test_from_list(self):
        ds = Dataset.from_list([{"input": "a"}, {"input": "b"}])
        assert len(ds) == 2
        assert ds[0]["input"] == "a"

    def test_from_jsonl(self, temp_dir):
        path = temp_dir / "test.jsonl"
        path.write_text('{"input": "a", "expected": "x"}\n{"input": "b", "expected": "y"}\n')
        ds = Dataset.from_jsonl(path)
        assert len(ds) == 2

    def test_iteration(self):
        ds = Dataset.from_list([{"x": 1}, {"x": 2}])
        items = list(ds)
        assert len(items) == 2


class TestScorers:
    def test_exact_match_pass(self):
        s = ExactMatch()
        r = s.score(input="q", output="answer", expected="answer")
        assert r.passed is True
        assert r.score == 1.0

    def test_exact_match_fail(self):
        s = ExactMatch()
        r = s.score(input="q", output="wrong", expected="right")
        assert r.passed is False

    def test_contains_pass(self):
        s = Contains()
        r = s.score(input="q", output="The answer is 42", expected="42")
        assert r.passed is True

    def test_contains_fail(self):
        s = Contains()
        r = s.score(input="q", output="No match here", expected="42")
        assert r.passed is False

    def test_json_valid_pass(self):
        s = JSONValid()
        r = s.score(input="q", output='{"key": "value"}')
        assert r.passed is True

    def test_json_valid_fail(self):
        s = JSONValid()
        r = s.score(input="q", output="not json")
        assert r.passed is False

    def test_length_between(self):
        s = LengthBetween(min_len=5, max_len=20)
        r = s.score(input="q", output="hello world")
        assert r.passed is True

        r2 = s.score(input="q", output="hi")
        assert r2.passed is False

    def test_code_scorer_decorator(self):
        @Scorer.code("custom")
        def check(input, output, expected=None):
            return len(output) > 3

        r = check.score(input="q", output="hello")
        assert r.passed is True
        assert r.score == 1.0


class TestTrajectoryScorers:
    def test_tool_usage_accuracy(self):
        s = ToolUsageAccuracy()
        r = s.score(
            input="", output="",
            actual_trajectory=["search", "calculate"],
            expected_trajectory=["search", "calculate", "format"],
        )
        assert r.score == pytest.approx(2 / 3)

    def test_step_efficiency(self):
        s = StepEfficiency()
        r = s.score(input="", output="", actual_steps=6, expected_steps=3)
        assert r.score == pytest.approx(0.5)

    def test_path_correctness(self):
        s = PathCorrectness()
        r = s.score(
            input="", output="",
            actual_trajectory=["a", "b", "c", "d"],
            expected_trajectory=["a", "c", "d"],
        )
        assert r.score == pytest.approx(1.0)

    def test_cycle_efficiency(self):
        s = CycleEfficiency()
        r = s.score(
            input="", output="",
            actual_trajectory=["a", "a", "b", "b", "c"],
        )
        assert r.score < 1.0  # has repeated consecutive calls


class TestEvalResults:
    def test_summary(self):
        results = EvalResults()
        results.add("exact_match", ScorerResult(score=1.0, passed=True))
        results.add("exact_match", ScorerResult(score=0.0, passed=False))
        summary = results.summary()
        assert "exact_match" in summary
        assert "50%" in summary

    def test_export(self, temp_dir):
        results = EvalResults()
        results.add("test", ScorerResult(score=0.8, passed=True))
        path = temp_dir / "results.json"
        results.export(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "test" in data

    def test_compare(self):
        a = EvalResults()
        a.add("s", ScorerResult(score=0.5, passed=True))
        b = EvalResults()
        b.add("s", ScorerResult(score=0.8, passed=True))
        diff = a.compare(b)
        assert "+0.30" in diff


class TestEvaluate:
    def test_evaluate_simple(self):
        def agent_fn(input_text):
            return input_text.upper()

        results = evaluate(
            agent_fn=agent_fn,
            dataset=[
                {"input": "hello", "expected": "HELLO"},
                {"input": "world", "expected": "WORLD"},
            ],
            scorers=["exact_match"],
        )
        summary = results.summary()
        assert "exact_match" in summary
        # Both should pass since upper() matches expected
        scores = results.scores["exact_match"]
        assert all(s.passed for s in scores)

    def test_evaluate_with_custom_scorer(self):
        @Scorer.code("is_upper")
        def check(input, output, expected=None):
            return output == output.upper()

        results = evaluate(
            agent_fn=lambda x: x.upper(),
            dataset=[{"input": "test"}],
            scorers=[check],
        )
        assert results.scores["is_upper"][0].passed is True
