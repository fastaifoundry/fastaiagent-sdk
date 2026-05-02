"""Evaluation framework with scorers, LLM judge, and dataset support."""

from fastaiagent.eval.dataset import Dataset
from fastaiagent.eval.evaluate import evaluate
from fastaiagent.eval.llm_judge import LLMJudge
from fastaiagent.eval.rag import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from fastaiagent.eval.results import EvalResults
from fastaiagent.eval.safety import Bias, PIILeakage, Toxicity
from fastaiagent.eval.scorer import Scorer, ScorerResult
from fastaiagent.eval.similarity import (
    BLEUScore,
    LevenshteinDistance,
    ROUGEScore,
    SemanticSimilarity,
)

__all__ = [
    # Core
    "evaluate",
    "Dataset",
    "Scorer",
    "ScorerResult",
    "EvalResults",
    "LLMJudge",
    # RAG
    "Faithfulness",
    "AnswerRelevancy",
    "ContextPrecision",
    "ContextRecall",
    # Safety
    "Toxicity",
    "Bias",
    "PIILeakage",
    # Similarity & NLP
    "SemanticSimilarity",
    "BLEUScore",
    "ROUGEScore",
    "LevenshteinDistance",
]
