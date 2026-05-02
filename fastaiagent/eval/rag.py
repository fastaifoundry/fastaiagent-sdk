"""RAG evaluation scorers — faithfulness, relevancy, context precision/recall."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from fastaiagent._internal.async_utils import run_sync
from fastaiagent.eval.scorer import Scorer, ScorerResult

logger = logging.getLogger(__name__)


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences that LLMs sometimes wrap around JSON."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _resolve_context(kw: dict[str, Any]) -> str | None:
    """Extract context string from kwargs (supports 'context' or 'contexts')."""
    ctx = kw.get("context") or kw.get("contexts")
    if ctx is None:
        return None
    if isinstance(ctx, list):
        return "\n\n".join(str(c) for c in ctx)
    return str(ctx)


def _resolve_contexts_list(kw: dict[str, Any]) -> list[str] | None:
    """Extract context as an ordered list of chunks."""
    ctx = kw.get("contexts")
    if isinstance(ctx, list):
        return [str(c) for c in ctx]
    ctx_str = kw.get("context")
    if ctx_str is not None:
        return [str(ctx_str)]
    return None


class Faithfulness(Scorer):
    """Measures factual consistency of the response with retrieved context.

    Two-step LLM process:
    1. Extract claims from the output.
    2. Verify each claim against the provided context.

    Score = supported claims / total claims.

    Pass context via kwargs: ``context="..."`` or ``contexts=["chunk1", "chunk2"]``.

    Example:
        scorer = Faithfulness()
        result = scorer.score(
            input="What is Python?",
            output="Python is a programming language created by Guido van Rossum.",
            context="Python is a programming language. It was created by Guido van Rossum in 1991.",
        )
    """

    name = "faithfulness"

    def __init__(self, llm: Any = None, threshold: float = 0.7):
        self._llm = llm
        self.threshold = threshold

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        return run_sync(self.ascore(input, output, expected, **kw))

    async def ascore(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        from fastaiagent.llm import LLMClient, SystemMessage, UserMessage

        context = _resolve_context(kw)
        if not context:
            return ScorerResult(score=0.0, passed=False, reason="No context provided")

        llm = self._llm or LLMClient()

        # Step 1: Extract claims
        try:
            extract_resp = await llm.acomplete(
                [
                    SystemMessage("You are a claim extraction assistant. Respond with JSON only."),
                    UserMessage(
                        "Break the following response into individual factual claims.\n"
                        "Return a JSON object with a 'claims' key containing a list of strings.\n\n"
                        f"Response: {output}\n\n"
                        'Example: {{"claims": ["claim 1", "claim 2"]}}'
                    ),
                ]
            )
            raw = _strip_code_fences(extract_resp.content or "")
            claims_data = json.loads(raw)
            claims = claims_data.get("claims", [])
            if not claims:
                return ScorerResult(score=1.0, passed=True, reason="No claims extracted")
        except Exception as e:
            return ScorerResult(score=0.0, passed=False, reason=f"Claim extraction error: {e}")

        # Step 2: Verify each claim against context
        supported = 0
        for claim in claims:
            try:
                verify_resp = await llm.acomplete(
                    [
                        SystemMessage("You are a fact-checking assistant. Respond with JSON only."),
                        UserMessage(
                            "Determine if the following claim is supported by "
                            "the given context.\n\n"
                            f"Context: {context}\n\n"
                            f"Claim: {claim}\n\n"
                            'Respond with JSON: {{"supported": true/false, "reasoning": "..."}}'
                        ),
                    ]
                )
                raw = _strip_code_fences(verify_resp.content or "")
                verdict = json.loads(raw)
                if verdict.get("supported", False):
                    supported += 1
            except Exception:
                logger.debug("Failed to verify claim in faithfulness scorer", exc_info=True)
                continue

        score_val = supported / len(claims)
        return ScorerResult(
            score=round(score_val, 4),
            passed=score_val >= self.threshold,
            reason=f"Supported claims: {supported}/{len(claims)}",
        )


class AnswerRelevancy(Scorer):
    """Measures how relevant the response is to the user's query.

    Single LLM call — does not require ``expected`` or ``context``.

    Example:
        scorer = AnswerRelevancy()
        result = scorer.score(
            input="What is the capital of France?",
            output="Paris is the capital of France.",
        )
    """

    name = "answer_relevancy"

    def __init__(self, llm: Any = None, threshold: float = 0.7):
        self._llm = llm
        self.threshold = threshold

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        return run_sync(self.ascore(input, output, expected, **kw))

    async def ascore(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        from fastaiagent.llm import LLMClient, SystemMessage, UserMessage

        llm = self._llm or LLMClient()

        prompt = (
            "Rate how relevant the following response is to the given question.\n\n"
            f"Question: {input}\n"
            f"Response: {output}\n\n"
            "Score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).\n\n"
            'Respond with JSON only: {{"score": <0.0-1.0>, "reasoning": "<explanation>"}}'
        )

        try:
            response = await llm.acomplete(
                [
                    SystemMessage("You are a relevancy evaluator. Respond with JSON only."),
                    UserMessage(prompt),
                ]
            )
            raw = _strip_code_fences(response.content or "")
            data = json.loads(raw)
            score_val = float(data.get("score", 0))
            reasoning = data.get("reasoning", "")
            return ScorerResult(
                score=score_val,
                passed=score_val >= self.threshold,
                reason=reasoning,
            )
        except Exception as e:
            return ScorerResult(score=0.0, passed=False, reason=f"Relevancy check error: {e}")


class ContextPrecision(Scorer):
    """Measures whether relevant context chunks are ranked higher.

    Requires ``contexts`` as an ordered list in kwargs (retrieval rank order).
    Uses Average Precision: rewards relevant documents appearing earlier.

    Example:
        scorer = ContextPrecision()
        result = scorer.score(
            input="What is Python?",
            output="Python is a programming language.",
            contexts=[
                "Python is a programming language created by Guido.",
                "Java is a popular enterprise language.",
                "Python supports multiple paradigms.",
            ],
        )
    """

    name = "context_precision"

    def __init__(self, llm: Any = None, threshold: float = 0.5):
        self._llm = llm
        self.threshold = threshold

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        return run_sync(self.ascore(input, output, expected, **kw))

    async def ascore(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        from fastaiagent.llm import LLMClient, SystemMessage, UserMessage

        chunks = _resolve_contexts_list(kw)
        if not chunks:
            return ScorerResult(score=0.0, passed=False, reason="No contexts provided")

        llm = self._llm or LLMClient()

        # Judge each chunk's relevance
        relevance: list[bool] = []
        for chunk in chunks:
            try:
                resp = await llm.acomplete(
                    [
                        SystemMessage("You are a relevance evaluator. Respond with JSON only."),
                        UserMessage(
                            "Is this context chunk relevant to answering the question?\n\n"
                            f"Question: {input}\n\n"
                            f"Context chunk: {chunk}\n\n"
                            'Respond with JSON: {{"relevant": true/false}}'
                        ),
                    ]
                )
                raw = _strip_code_fences(resp.content or "")
                data = json.loads(raw)
                relevance.append(bool(data.get("relevant", False)))
            except Exception:
                logger.debug("Failed to evaluate context relevance for chunk", exc_info=True)
                relevance.append(False)

        # Average Precision
        total_relevant = sum(relevance)
        if total_relevant == 0:
            return ScorerResult(
                score=0.0,
                passed=False,
                reason="No relevant context chunks found",
            )

        ap = 0.0
        relevant_so_far = 0
        for k, is_rel in enumerate(relevance, 1):
            if is_rel:
                relevant_so_far += 1
                ap += relevant_so_far / k

        score_val = ap / total_relevant

        return ScorerResult(
            score=round(score_val, 4),
            passed=score_val >= self.threshold,
            reason=f"AP={score_val:.4f} ({total_relevant}/{len(chunks)} relevant)",
        )


class ContextRecall(Scorer):
    """Measures what fraction of the expected answer's claims appear in the context.

    Requires both ``expected`` and ``context``/``contexts`` in kwargs.

    Example:
        scorer = ContextRecall()
        result = scorer.score(
            input="What is Python?",
            output="anything",
            expected="Python is a programming language created by Guido van Rossum in 1991.",
            context="Python is a programming language. Guido van Rossum created it.",
        )
    """

    name = "context_recall"

    def __init__(self, llm: Any = None, threshold: float = 0.5):
        self._llm = llm
        self.threshold = threshold

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        return run_sync(self.ascore(input, output, expected, **kw))

    async def ascore(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        from fastaiagent.llm import LLMClient, SystemMessage, UserMessage

        context = _resolve_context(kw)
        if not context:
            return ScorerResult(score=0.0, passed=False, reason="No context provided")
        if not expected:
            return ScorerResult(score=0.0, passed=False, reason="No expected output provided")

        llm = self._llm or LLMClient()

        # Step 1: Extract claims from expected answer
        try:
            extract_resp = await llm.acomplete(
                [
                    SystemMessage("You are a claim extraction assistant. Respond with JSON only."),
                    UserMessage(
                        "Break the following reference answer into individual factual claims.\n"
                        "Return a JSON object with a 'claims' key containing a list of strings.\n\n"
                        f"Reference: {expected}\n\n"
                        'Example: {{"claims": ["claim 1", "claim 2"]}}'
                    ),
                ]
            )
            raw = _strip_code_fences(extract_resp.content or "")
            claims_data = json.loads(raw)
            claims = claims_data.get("claims", [])
            if not claims:
                return ScorerResult(score=1.0, passed=True, reason="No claims in expected")
        except Exception as e:
            return ScorerResult(score=0.0, passed=False, reason=f"Claim extraction error: {e}")

        # Step 2: Check each claim against context
        found = 0
        for claim in claims:
            try:
                resp = await llm.acomplete(
                    [
                        SystemMessage("You are a fact-checking assistant. Respond with JSON only."),
                        UserMessage(
                            "Is the following claim present or supported in the given context?\n\n"
                            f"Context: {context}\n\n"
                            f"Claim: {claim}\n\n"
                            'Respond with JSON: {{"present": true/false}}'
                        ),
                    ]
                )
                raw = _strip_code_fences(resp.content or "")
                data = json.loads(raw)
                if data.get("present", False):
                    found += 1
            except Exception:
                logger.debug(
                    "Failed to check claim presence in context recall scorer", exc_info=True,
                )
                continue

        score_val = found / len(claims)
        return ScorerResult(
            score=round(score_val, 4),
            passed=score_val >= self.threshold,
            reason=f"Claims in context: {found}/{len(claims)}",
        )
