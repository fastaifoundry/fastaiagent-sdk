"""Every scorer reports could-not-run as **errored** and does-not-apply as **skipped**.

The eval twin of ``test_guardrail_unusable_config_sweep.py`` (CLAUDE.md §2.4):
before 1.89.0 a judge outage read as a failed case, an unparseable verdict as a
score of 0.0, a ``{}`` reply as "no injection detected" / "no claims", and a
missing expectation as a pass. None of those is a verdict.

Three axes per scorer:

* **outage**: the model / embedder / moderation client raises;
* **empty verdict**: the judge replies ``{}``, valid JSON with no verdict field;
* **missing input**: the case lacks what the scorer needs.

``test_every_scorer_is_in_the_sweep`` fails when a scorer class is added without
a row here; the exemptions say why a scorer has no such path. No mocks: real
scorers, real ``FunctionModel`` clients.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any

import pytest

import fastaiagent.eval as ev
from fastaiagent.eval import (
    agent_metrics,
    decision_judge,
    llm_judge,
    rag,
    safety,
    session,
    similarity,
    trajectory,
)
from fastaiagent.eval import builtins as core
from fastaiagent.eval.builtins import BUILTIN_SCORERS
from fastaiagent.eval.scorer import CodeScorer, Scorer, ScorerResult
from fastaiagent.testing import FunctionModel


def _boom(messages: Any) -> str:
    raise RuntimeError("provider down")


def _raising_decide(input: Any, questions: Any) -> Any:
    raise RuntimeError("provider down")


def down() -> FunctionModel:
    return FunctionModel(_boom, decide_fn=_raising_decide)


def empty() -> FunctionModel:
    return FunctionModel(lambda messages: "{}")


class _DownEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedder down")


class _DownModeration:
    class moderations:  # noqa: N801 - mirrors the OpenAI client shape
        @staticmethod
        def create(**kwargs: Any) -> Any:
            raise RuntimeError("moderation down")


FULL: dict[str, Any] = {
    "input": "What is the refund window?",
    "output": "Refunds are accepted within 30 days.",
    "expected": "30 days",
    "context": "Our policy: refunds within 30 days of purchase.",
    "contexts": ["Our policy: refunds within 30 days of purchase."],
    "turns": [
        {"role": "user", "content": "Can I get a refund?"},
        {"role": "assistant", "content": "Yes, within 30 days."},
    ],
    "goal": "explain the refund window",
}

# --------------------------------------------------------------------------- #
# Axis 1: outage -> errored
# --------------------------------------------------------------------------- #

OUTAGE: dict[str, Callable[[], Scorer]] = {
    "Hallucination": lambda: agent_metrics.Hallucination(llm=down()),
    "ReflectionQuality": lambda: agent_metrics.ReflectionQuality(llm=down()),
    "TaskCompletion": lambda: agent_metrics.TaskCompletion(llm=down()),
    "DecisionJudge": lambda: decision_judge.DecisionJudge("correct?", llm=down()),
    "GEval": lambda: llm_judge.GEval(llm=down(), evaluation_steps=["check"]),
    "LLMJudge": lambda: llm_judge.LLMJudge(llm=down()),
    "AnswerRelevancy": lambda: rag.AnswerRelevancy(llm=down()),
    "ContextPrecision": lambda: rag.ContextPrecision(llm=down()),
    "ContextRecall": lambda: rag.ContextRecall(llm=down()),
    "Faithfulness": lambda: rag.Faithfulness(llm=down()),
    "Bias": lambda: safety.Bias(llm=down()),
    "OpenAIModeration": lambda: safety.OpenAIModeration(client=_DownModeration()),
    "PromptInjection": lambda: safety.PromptInjection(mode="llm", llm=down()),
    "Toxicity": lambda: safety.Toxicity(llm=down()),
    "ConversationCoherence": lambda: session.ConversationCoherence(mode="llm", llm=down()),
    "ConversationRelevancy": lambda: session.ConversationRelevancy(llm=down()),
    "GoalCompletion": lambda: session.GoalCompletion(mode="llm", llm=down()),
    "KnowledgeRetention": lambda: session.KnowledgeRetention(llm=down()),
    "RoleAdherence": lambda: session.RoleAdherence(role="support agent", llm=down()),
    "SemanticSimilarity": lambda: similarity.SemanticSimilarity(embedder=_DownEmbedder()),
}

#: No external dependency to lose: these compute their verdict locally.
NO_OUTAGE_PATH: dict[str, str] = {
    "Contains": "string containment",
    "CostUnder": "reads the run's cost; an unpriced run is the unknown-cost cell below",
    "ExactMatch": "string equality",
    "JSONValid": "json.loads",
    "Latency": "reads the run's latency",
    "LengthBetween": "len()",
    "RegexMatch": "re.search",
    "PIILeakage": "regex detectors (the presidio backend is chosen at construction)",
    "CodeScorer": "the user's function; a raise is the score_case guard (test_eval_scorer_guard)",
    "BLEUScore": "n-gram arithmetic",
    "LevenshteinDistance": "edit distance",
    "ROUGEScore": "n-gram arithmetic",
    "CycleEfficiency": "reads the recorded trajectory",
    "PathCorrectness": "reads the recorded trajectory",
    "StepEfficiency": "reads the recorded trajectory",
    "ToolCallCorrectness": "reads the recorded tool calls",
    "ToolUsageAccuracy": "reads the recorded trajectory",
}

# --------------------------------------------------------------------------- #
# Axis 2: a ``{}`` reply -> errored (the judge gave no verdict)
# --------------------------------------------------------------------------- #

EMPTY_VERDICT: dict[str, Callable[[], Scorer]] = {
    "Hallucination": lambda: agent_metrics.Hallucination(llm=empty()),
    "ReflectionQuality": lambda: agent_metrics.ReflectionQuality(llm=empty()),
    "TaskCompletion": lambda: agent_metrics.TaskCompletion(llm=empty()),
    "GEval": lambda: llm_judge.GEval(llm=empty(), evaluation_steps=["check"]),
    "LLMJudge": lambda: llm_judge.LLMJudge(llm=empty()),
    "AnswerRelevancy": lambda: rag.AnswerRelevancy(llm=empty()),
    "ContextPrecision": lambda: rag.ContextPrecision(llm=empty()),
    "ContextRecall": lambda: rag.ContextRecall(llm=empty()),
    "Faithfulness": lambda: rag.Faithfulness(llm=empty()),
    "Bias": lambda: safety.Bias(llm=empty()),
    "PromptInjection": lambda: safety.PromptInjection(mode="llm", llm=empty()),
    "Toxicity": lambda: safety.Toxicity(llm=empty()),
    "ConversationCoherence": lambda: session.ConversationCoherence(mode="llm", llm=empty()),
    "ConversationRelevancy": lambda: session.ConversationRelevancy(llm=empty()),
    "GoalCompletion": lambda: session.GoalCompletion(mode="llm", llm=empty()),
    "KnowledgeRetention": lambda: session.KnowledgeRetention(llm=empty()),
    "RoleAdherence": lambda: session.RoleAdherence(role="support agent", llm=empty()),
}

NO_JSON_VERDICT: dict[str, str] = {
    **NO_OUTAGE_PATH,
    "DecisionJudge": "structured Decisions API answer, not free JSON",
    "OpenAIModeration": "the moderation endpoint's typed response",
    "SemanticSimilarity": "vectors, not a judge reply",
}

# --------------------------------------------------------------------------- #
# Axis 3: missing input -> skipped
# --------------------------------------------------------------------------- #

BARE: dict[str, Any] = {"input": "q", "output": "a"}

MISSING_INPUT: dict[str, tuple[Callable[[], Scorer], dict[str, Any]]] = {
    "Hallucination": (lambda: agent_metrics.Hallucination(llm=down()), BARE),
    "Contains": (core.Contains, BARE),
    "ExactMatch": (core.ExactMatch, BARE),
    "CostUnder": (core.CostUnder, BARE),
    "Latency": (core.Latency, BARE),
    "ContextPrecision": (lambda: rag.ContextPrecision(llm=down()), BARE),
    "ContextRecall": (lambda: rag.ContextRecall(llm=down()), BARE),
    "Faithfulness": (lambda: rag.Faithfulness(llm=down()), BARE),
    "ConversationCoherence": (
        lambda: session.ConversationCoherence(mode="llm", llm=down()),
        BARE,
    ),
    "ConversationRelevancy": (lambda: session.ConversationRelevancy(llm=down()), BARE),
    "GoalCompletion": (lambda: session.GoalCompletion(mode="llm", llm=down()), BARE),
    "KnowledgeRetention": (lambda: session.KnowledgeRetention(llm=down()), BARE),
    "RoleAdherence": (lambda: session.RoleAdherence(llm=down()), {**BARE, "turns": FULL["turns"]}),
    "BLEUScore": (similarity.BLEUScore, BARE),
    "LevenshteinDistance": (similarity.LevenshteinDistance, BARE),
    "ROUGEScore": (similarity.ROUGEScore, BARE),
    "SemanticSimilarity": (lambda: similarity.SemanticSimilarity(embedder=_DownEmbedder()), BARE),
    "PathCorrectness": (trajectory.PathCorrectness, {**BARE, "actual_trajectory": ["a"]}),
    "StepEfficiency": (trajectory.StepEfficiency, {**BARE, "actual_steps": 1}),
    "ToolCallCorrectness": (
        trajectory.ToolCallCorrectness,
        {**BARE, "actual_tool_calls": [{"name": "a", "arguments": {}}]},
    ),
    "ToolUsageAccuracy": (trajectory.ToolUsageAccuracy, {**BARE, "actual_trajectory": ["a"]}),
}

#: Every case carries what these read (input and output are always present).
NO_MISSING_INPUT: dict[str, str] = {
    "JSONValid": "reads output only",
    "LengthBetween": "reads output only",
    "RegexMatch": "reads output only",
    "PIILeakage": "reads output only",
    "CodeScorer": "the user's function decides",
    "CycleEfficiency": "needs no expectation; a missing record is errored (trajectory cells)",
    "ReflectionQuality": "reads input and output only",
    "TaskCompletion": "reads input and output only",
    "DecisionJudge": "reads input and output only",
    "GEval": "reads input and output only",
    "LLMJudge": "reads input and output only",
    "AnswerRelevancy": "reads input and output only",
    "Bias": "reads output only",
    "OpenAIModeration": "reads output only",
    "PromptInjection": "reads input only",
    "Toxicity": "reads output only",
}


def _all_scorer_classes() -> set[str]:
    for mod in pkgutil.iter_modules(ev.__path__):
        importlib.import_module(f"fastaiagent.eval.{mod.name}")

    def walk(cls: type) -> set[type]:
        out: set[type] = set()
        for sub in cls.__subclasses__():
            if sub.__module__.startswith("fastaiagent."):
                out.add(sub)
            out |= walk(sub)
        return out

    return {c.__name__ for c in walk(Scorer)}


def _assert_not_a_verdict(result: ScorerResult, outcome: str) -> None:
    assert result.outcome == outcome, (result.outcome, result.reason)
    assert result.passed is False and result.score == 0.0
    assert result.reason, "a could-not-run / not-applicable result must say why"


def test_every_scorer_is_in_the_sweep() -> None:
    classes = _all_scorer_classes()
    assert classes, "found no Scorer subclasses"
    for axis, covered in (
        ("outage", set(OUTAGE) | set(NO_OUTAGE_PATH)),
        ("empty verdict", set(EMPTY_VERDICT) | set(NO_JSON_VERDICT)),
        ("missing input", set(MISSING_INPUT) | set(NO_MISSING_INPUT)),
    ):
        missing = classes - covered
        assert not missing, f"{axis}: add a sweep row or an exemption for {sorted(missing)}"
        stale = covered - classes
        assert not stale, f"{axis}: rows for scorers that no longer exist: {sorted(stale)}"


def test_every_registered_name_is_a_swept_class() -> None:
    swept = _all_scorer_classes()
    for name, cls in BUILTIN_SCORERS.items():
        assert cls.__name__ in swept, name


@pytest.mark.parametrize("name", sorted(OUTAGE))
def test_outage_is_errored(name: str) -> None:
    _assert_not_a_verdict(OUTAGE[name]().score(**FULL), "errored")


@pytest.mark.parametrize("name", sorted(EMPTY_VERDICT))
def test_reply_without_a_verdict_is_errored(name: str) -> None:
    """``{}`` used to read as the field's default: "no injection", "no claims", 0.0."""
    _assert_not_a_verdict(EMPTY_VERDICT[name]().score(**FULL), "errored")


@pytest.mark.parametrize("name", sorted(MISSING_INPUT))
def test_missing_input_is_skipped(name: str) -> None:
    make, kwargs = MISSING_INPUT[name]
    _assert_not_a_verdict(make().score(**kwargs), "skipped")


def test_unpriced_run_is_errored_not_free() -> None:
    r = core.CostUnder(max_usd=1.0).score(**BARE, cost=0.0, cost_known=False)
    _assert_not_a_verdict(r, "errored")
    assert "unknown" in (r.reason or "")


@pytest.mark.parametrize(
    "scorer",
    [
        trajectory.ToolUsageAccuracy(),
        trajectory.PathCorrectness(),
        trajectory.ToolCallCorrectness(),
        trajectory.StepEfficiency(),
    ],
    ids=lambda s: type(s).__name__,
)
def test_trajectory_with_expectation_but_no_record_is_errored(scorer: Scorer) -> None:
    """A plain-function agent returns a string: there is no record to compare."""
    kwargs = {
        "expected_trajectory": ["lookup"],
        "expected_tool_calls": [{"name": "lookup", "arguments": {}}],
        "expected_steps": 1,
    }
    _assert_not_a_verdict(scorer.score(**BARE, **kwargs), "errored")


def test_cycle_efficiency_without_a_record_is_errored() -> None:
    _assert_not_a_verdict(trajectory.CycleEfficiency().score(**BARE), "errored")


def test_a_verdict_still_is_one() -> None:
    """The control: the same scorers still judge when they can."""
    ok = FunctionModel(lambda messages: '{"score": 0.9, "reasoning": "fine"}')
    r = agent_metrics.TaskCompletion(llm=ok).score(**FULL)
    assert r.outcome == "passed" and r.score == 0.9
    r = core.ExactMatch().score(input="q", output="a", expected="b")
    assert r.outcome == "failed"
    r = trajectory.ToolUsageAccuracy().score(
        **BARE, actual_trajectory=[], expected_trajectory=["lookup"]
    )
    assert r.outcome == "failed", "an empty record against an expectation is a real fail"


def test_code_scorer_is_swept_as_a_class() -> None:
    assert isinstance(CodeScorer("c", lambda input, output, expected=None: True), Scorer)
