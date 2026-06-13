"""Evaluation framework with scorers, LLM judge, and dataset support."""

from typing import Any

from fastaiagent.eval.dataset import Dataset
from fastaiagent.eval.evaluate import evaluate
from fastaiagent.eval.llm_judge import LLMJudge

# Pytest plugin decorators are an *optional* surface — the eval
# framework itself works without pytest, but the ``@case`` / ``@dataset``
# decorators below need it. Older versions imported the plugin
# unconditionally, which made plain ``import fastaiagent`` raise
# ``ModuleNotFoundError: No module named 'pytest'`` on any install
# without pytest available (the typical production server).
# We now expose stubs that raise a helpful ImportError only if the
# decorator is actually called.
try:
    from fastaiagent.eval.pytest_plugin import case
    from fastaiagent.eval.pytest_plugin import dataset as pytest_dataset
except ImportError:  # pragma: no cover — exercised by subprocess test

    def _missing_pytest(*_args: Any, **_kwargs: Any) -> Any:
        raise ImportError(
            "fastaiagent.eval.case / fastaiagent.eval.pytest_dataset require "
            "``pytest``. Install it with `pip install pytest` or include the "
            "fastaiagent[testing] extra in your dev environment."
        )

    case = _missing_pytest  # type: ignore[assignment]
    pytest_dataset = _missing_pytest  # type: ignore[assignment]

from fastaiagent.eval.agent_metrics import Hallucination, ReflectionQuality, TaskCompletion
from fastaiagent.eval.harden import HardeningReport, Recommendation, aharden, harden
from fastaiagent.eval.rag import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from fastaiagent.eval.results import EvalResults, MetricSummary, Scorecard
from fastaiagent.eval.safety import (
    Bias,
    OpenAIModeration,
    PIILeakage,
    PromptInjection,
    Toxicity,
)
from fastaiagent.eval.scorer import Scorer, ScorerResult
from fastaiagent.eval.similarity import (
    BLEUScore,
    LevenshteinDistance,
    ROUGEScore,
    SemanticSimilarity,
)
from fastaiagent.eval.simulate import (
    Scenario,
    SimulatedUser,
    SimulationResult,
    SimulationResults,
    agenerate_scenarios,
    asimulate,
    generate_scenarios,
    simulate,
)

__all__ = [
    # Core
    "evaluate",
    "Dataset",
    "Scorer",
    "ScorerResult",
    "EvalResults",
    "LLMJudge",
    "Scorecard",
    "MetricSummary",
    # Simulation
    "simulate",
    "asimulate",
    "Scenario",
    "SimulatedUser",
    "SimulationResult",
    "SimulationResults",
    "generate_scenarios",
    "agenerate_scenarios",
    # Agent hardening
    "harden",
    "aharden",
    "HardeningReport",
    "Recommendation",
    # Agent-eval metrics
    "TaskCompletion",
    "Hallucination",
    "ReflectionQuality",
    # Pytest plugin decorators
    "case",
    "pytest_dataset",
    # RAG
    "Faithfulness",
    "AnswerRelevancy",
    "ContextPrecision",
    "ContextRecall",
    # Safety
    "Toxicity",
    "Bias",
    "PIILeakage",
    "PromptInjection",
    "OpenAIModeration",
    # Similarity & NLP
    "SemanticSimilarity",
    "BLEUScore",
    "ROUGEScore",
    "LevenshteinDistance",
]
