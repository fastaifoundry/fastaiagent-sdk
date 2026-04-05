"""Tests for fastaiagent.eval module."""

from __future__ import annotations

import json

import pytest

from fastaiagent.eval import Dataset, EvalResults, Scorer, ScorerResult, evaluate
from fastaiagent.eval.builtins import Contains, ExactMatch, JSONValid, LengthBetween
from fastaiagent.eval.safety import PIILeakage
from fastaiagent.eval.similarity import BLEUScore, LevenshteinDistance, ROUGEScore
from fastaiagent.eval.trajectory import (
    CycleEfficiency,
    PathCorrectness,
    StepEfficiency,
    ToolCallCorrectness,
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
            input="",
            output="",
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
            input="",
            output="",
            actual_trajectory=["a", "b", "c", "d"],
            expected_trajectory=["a", "c", "d"],
        )
        assert r.score == pytest.approx(1.0)

    def test_cycle_efficiency(self):
        s = CycleEfficiency()
        r = s.score(
            input="",
            output="",
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


class TestSimilarityScorers:
    def test_bleu_perfect_match(self):
        s = BLEUScore()
        r = s.score(input="q", output="the cat sat on the mat", expected="the cat sat on the mat")
        assert r.score > 0.9
        assert r.passed is True

    def test_bleu_no_overlap(self):
        s = BLEUScore()
        r = s.score(input="q", output="completely different words here", expected="nothing matches at all")
        assert r.score == 0.0
        assert r.passed is False

    def test_bleu_partial_match(self):
        s = BLEUScore(max_n=2)
        r = s.score(input="q", output="the cat sat on the mat", expected="the cat is on the mat")
        assert 0.0 < r.score < 1.0

    def test_bleu_no_expected(self):
        s = BLEUScore()
        r = s.score(input="q", output="hello")
        assert r.passed is False

    def test_rouge_1_perfect(self):
        s = ROUGEScore(variant="rouge-1")
        r = s.score(input="q", output="the cat sat on mat", expected="the cat sat on mat")
        assert r.score == pytest.approx(1.0)

    def test_rouge_1_partial(self):
        s = ROUGEScore(variant="rouge-1")
        r = s.score(input="q", output="the cat sat on the mat", expected="the cat is on the mat")
        assert 0.0 < r.score < 1.0

    def test_rouge_l_subsequence(self):
        s = ROUGEScore(variant="rouge-l")
        r = s.score(input="q", output="the cat sat on the mat", expected="the cat is on the mat")
        assert r.score > 0.5

    def test_rouge_no_overlap(self):
        s = ROUGEScore(variant="rouge-1")
        r = s.score(input="q", output="alpha beta gamma", expected="delta epsilon zeta")
        assert r.score == 0.0

    def test_levenshtein_identical(self):
        s = LevenshteinDistance()
        r = s.score(input="q", output="hello world", expected="hello world")
        assert r.score == pytest.approx(1.0)

    def test_levenshtein_completely_different(self):
        s = LevenshteinDistance()
        r = s.score(input="q", output="abc", expected="xyz")
        assert r.score == 0.0

    def test_levenshtein_one_edit(self):
        s = LevenshteinDistance()
        r = s.score(input="q", output="kitten", expected="sitten")
        assert r.score > 0.8

    def test_levenshtein_no_expected(self):
        s = LevenshteinDistance()
        r = s.score(input="q", output="hello")
        assert r.passed is False


class TestSafetyScorers:
    def test_pii_no_leak(self):
        s = PIILeakage()
        r = s.score(input="q", output="The weather is sunny today.")
        assert r.passed is True
        assert r.score == 1.0

    def test_pii_email_detected(self):
        s = PIILeakage()
        r = s.score(input="q", output="Contact me at john@example.com for details.")
        assert r.passed is False
        assert "email" in r.reason

    def test_pii_phone_detected(self):
        s = PIILeakage()
        r = s.score(input="q", output="Call me at 555-123-4567.")
        assert r.passed is False
        assert "phone" in r.reason

    def test_pii_ssn_detected(self):
        s = PIILeakage()
        r = s.score(input="q", output="My SSN is 123-45-6789.")
        assert r.passed is False
        assert "ssn" in r.reason

    def test_pii_credit_card_detected(self):
        s = PIILeakage()
        r = s.score(input="q", output="Card number: 4111 1111 1111 1111")
        assert r.passed is False
        assert "credit_card" in r.reason

    def test_pii_multiple_types(self):
        s = PIILeakage()
        r = s.score(
            input="q",
            output="Email john@test.com, SSN 123-45-6789, call 555-123-4567",
        )
        assert r.passed is False
        assert "email" in r.reason
        assert "ssn" in r.reason
        assert "phone" in r.reason


class TestToolCallCorrectness:
    def test_exact_match(self):
        s = ToolCallCorrectness()
        r = s.score(
            input="", output="",
            actual_tool_calls=[
                {"name": "search", "arguments": {"query": "Paris"}},
                {"name": "format", "arguments": {"style": "markdown"}},
            ],
            expected_tool_calls=[
                {"name": "search", "arguments": {"query": "Paris"}},
                {"name": "format", "arguments": {"style": "markdown"}},
            ],
        )
        assert r.score == pytest.approx(1.0)
        assert r.passed is True

    def test_partial_match(self):
        s = ToolCallCorrectness()
        r = s.score(
            input="", output="",
            actual_tool_calls=[
                {"name": "search", "arguments": {"query": "Paris"}},
            ],
            expected_tool_calls=[
                {"name": "search", "arguments": {"query": "Paris"}},
                {"name": "format", "arguments": {"style": "markdown"}},
            ],
        )
        assert r.score == pytest.approx(0.5)

    def test_wrong_args(self):
        s = ToolCallCorrectness()
        r = s.score(
            input="", output="",
            actual_tool_calls=[
                {"name": "search", "arguments": {"query": "London"}},
            ],
            expected_tool_calls=[
                {"name": "search", "arguments": {"query": "Paris"}},
            ],
        )
        assert r.score == 0.0

    def test_empty_expected(self):
        s = ToolCallCorrectness()
        r = s.score(
            input="", output="",
            actual_tool_calls=[{"name": "search", "arguments": {}}],
            expected_tool_calls=[],
        )
        assert r.score == 1.0
        assert r.passed is True
