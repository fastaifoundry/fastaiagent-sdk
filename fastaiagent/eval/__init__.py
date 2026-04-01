"""Evaluation framework with scorers, LLM judge, and dataset support."""

from fastaiagent.eval.dataset import Dataset
from fastaiagent.eval.evaluate import evaluate
from fastaiagent.eval.llm_judge import LLMJudge
from fastaiagent.eval.results import EvalResults
from fastaiagent.eval.scorer import Scorer, ScorerResult

__all__ = [
    "evaluate",
    "Dataset",
    "Scorer",
    "ScorerResult",
    "EvalResults",
    "LLMJudge",
]
