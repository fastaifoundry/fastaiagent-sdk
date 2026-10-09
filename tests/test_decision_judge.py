"""``DecisionJudge`` — an eval judge on the Decisions API (1.84.0).

No mocks: the judge's real ``LLMClient.adecide`` runs against the local
``/v1/decisions`` stand-in; ``evaluate()`` and ``simulate()`` run for real.
"""

from __future__ import annotations

import pytest

from fastaiagent.agent.agent import Agent
from fastaiagent.eval import DecisionJudge, evaluate
from fastaiagent.eval.simulate import Scenario, SimulatedUser, simulate
from fastaiagent.testing.models import TestModel

CORRECT = "The actual output answers the input correctly."


class TestPredicateJudge:
    def test_score_is_the_probability(self, decisions_stub) -> None:
        decisions_stub.probabilities["judgement"] = 0.82
        r = DecisionJudge(CORRECT, llm=decisions_stub.client()).score("2+2?", "4", "4")
        assert r.score == 0.82 and r.passed
        assert "p=0.82" in r.reason

        sent = decisions_stub.requests[-1]["body"]
        assert sent["questions"] == [
            {"type": "predicate", "instructions": CORRECT, "name": "judgement"}
        ]
        # The case reaches the endpoint as evidence, not as instructions.
        assert "Actual output:\n4" in sent["input"]
        assert "Expected output:\n4" in sent["input"]

    def test_threshold(self, decisions_stub) -> None:
        decisions_stub.probabilities["judgement"] = 0.6
        judge = DecisionJudge(CORRECT, llm=decisions_stub.client(), threshold=0.7)
        assert not judge.score("q", "a").passed

    def test_missing_expected_renders_na(self, decisions_stub) -> None:
        DecisionJudge(CORRECT, llm=decisions_stub.client()).score("q", "a")
        assert "Expected output:\nN/A" in decisions_stub.requests[-1]["body"]["input"]

    def test_custom_template_with_legacy_alias(self, decisions_stub) -> None:
        judge = DecisionJudge(
            CORRECT, llm=decisions_stub.client(), template="Q={{input}} A={output} E={expected_output}"
        )
        judge.score("q", "a", "e")
        assert decisions_stub.requests[-1]["body"]["input"] == "Q=q A=a E=e"


class TestScoreJudge:
    def test_score_is_the_normalized_level(self, decisions_stub) -> None:
        decisions_stub.levels["judgement"] = 1  # middle of three
        judge = DecisionJudge(
            "How correct is the actual output?",
            levels=["Wrong", "Partially correct", "Correct"],
            llm=decisions_stub.client(),
            threshold=0.75,
        )
        r = judge.score("q", "a")
        assert r.score == 0.5 and not r.passed
        assert "'Partially correct'" in r.reason
        assert decisions_stub.requests[-1]["body"]["questions"][0]["type"] == "score"


class TestNeverAFalsePass:
    def test_a_refusal_scores_zero(self, decisions_stub) -> None:
        decisions_stub.refuse.add("judgement")
        r = DecisionJudge(CORRECT, llm=decisions_stub.client()).score("q", "a")
        assert r.score == 0.0 and not r.passed
        assert "refused" in r.reason

    def test_a_failed_call_scores_zero(self, decisions_stub) -> None:
        decisions_stub.script = [(500, {"error": {"message": "down"}})]
        r = DecisionJudge(CORRECT, llm=decisions_stub.client()).score("q", "a")
        assert r.score == 0.0 and not r.passed
        assert "Decision judge error" in r.reason

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"criteria": "  "}, "non-empty"),
            ({"criteria": CORRECT, "levels": ["only"]}, "at least 2 levels"),
            ({"criteria": CORRECT, "threshold": 1.5}, "threshold"),
            ({"criteria": CORRECT, "template": "no placeholders"}, "placeholder"),
        ],
    )
    def test_a_judge_that_cannot_judge_fails_at_construction(self, kwargs, match) -> None:
        with pytest.raises(ValueError, match=match):
            DecisionJudge(**kwargs)


class TestInEvaluate:
    def test_evaluate(self, decisions_stub) -> None:
        decisions_stub.probabilities["judgement"] = 0.9
        results = evaluate(
            lambda text: text.upper(),
            [{"input": "a", "expected": "A"}, {"input": "b", "expected": "B"}],
            scorers=[DecisionJudge(CORRECT, llm=decisions_stub.client(), name="correct")],
            persist=False,
        )
        summary = results.summary()
        assert "correct" in summary
        assert len([r for r in decisions_stub.requests if r["path"].endswith("/decisions")]) == 2


class TestInSimulate:
    def test_success_and_failure_criteria(self, decisions_stub) -> None:
        # Success criterion holds (0.9); failure criterion: did the bad thing happen?
        decisions_stub.probabilities["judgement"] = 0.9
        agent = Agent(name="bot", llm=TestModel(response="Your refund is on its way."))
        scenario = Scenario(
            name="refund",
            user=SimulatedUser(script=["Where is my refund?"]),
            success_criteria=["The agent tells the user the refund status."],
            failure_criteria=["The agent asks for a credit card number."],
        )
        results = simulate(
            scenario, agent, judge=DecisionJudge(CORRECT, llm=decisions_stub.client()), persist=False
        )
        r = results.results[0]
        kinds = {v.kind: v for v in r.verdicts}
        assert kinds["success"].passed
        # p=0.9 that the failure condition occurred → that criterion fails.
        assert not kinds["failure"].passed
        assert not r.passed

        questions = [req["body"]["questions"][0] for req in decisions_stub.requests]
        assert questions[0]["instructions"] == "The agent tells the user the refund status."
        assert questions[1]["type"] == "predicate"
        assert questions[1]["instructions"].startswith("This undesirable condition occurred")
