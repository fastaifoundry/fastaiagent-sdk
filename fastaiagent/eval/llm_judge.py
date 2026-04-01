"""LLM Judge scorer for evaluation."""

from __future__ import annotations

import json
from typing import Any

from fastaiagent._internal.async_utils import run_sync
from fastaiagent.eval.scorer import Scorer, ScorerResult


class LLMJudge(Scorer):
    """Scores output using an LLM as a judge.

    Example:
        judge = LLMJudge(criteria="correctness")
        result = judge.score(input="What is 2+2?", output="4", expected="4")
    """

    name = "llm_judge"

    def __init__(
        self,
        criteria: str = "correctness",
        prompt_template: str | None = None,
        llm: Any = None,
        scale: str = "binary",  # "binary", "0-1", "1-5"
    ):
        self.criteria = criteria
        self.prompt_template = prompt_template or self._default_prompt()
        self._llm = llm
        self.scale = scale

    def _default_prompt(self) -> str:
        return (
            f"Evaluate the following response for {self.criteria}.\n\n"
            "Input: {input}\n"
            "Expected: {expected}\n"
            "Actual Output: {output}\n\n"
            'Respond with JSON: {{"score": <number>, "reasoning": "<explanation>"}}\n'
            "Score should be between 0 and 1."
        )

    def score(
        self, input: str, output: str, expected: str | None = None, **kwargs: Any
    ) -> ScorerResult:
        """Score using LLM judge (sync)."""
        return run_sync(self.ascore(input, output, expected, **kwargs))

    async def ascore(
        self, input: str, output: str, expected: str | None = None, **kwargs: Any
    ) -> ScorerResult:
        """Score using LLM judge (async)."""
        from fastaiagent.llm import LLMClient, SystemMessage, UserMessage

        llm = self._llm or LLMClient()

        prompt = self.prompt_template.replace("{input}", input)
        prompt = prompt.replace("{output}", output)
        prompt = prompt.replace("{expected}", expected or "N/A")

        try:
            response = await llm.acomplete(
                [
                    SystemMessage("You are an evaluation judge. Respond with JSON only."),
                    UserMessage(prompt),
                ]
            )
            content = response.content or ""

            # Parse JSON response
            data = json.loads(content)
            score_val = float(data.get("score", 0))
            reasoning = data.get("reasoning", "")
            passed = score_val >= 0.5

            return ScorerResult(score=score_val, passed=passed, reason=reasoning)
        except Exception as e:
            return ScorerResult(score=0.0, passed=False, reason=f"Judge error: {e}")
