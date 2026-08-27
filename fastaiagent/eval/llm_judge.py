"""LLM Judge scorer for evaluation.

``LLMJudge`` runs in one of two modes:

* **Single-call** (default) — ``LLMJudge(criteria="correctness")``: builds a prompt
  from ``criteria``/``prompt_template``, makes one LLM call, normalizes the verdict
  from ``scale`` to 0-1, and passes when ``score >= threshold``.
* **G-Eval** — opt in by passing ``evaluation_steps`` and/or a score-band ``rubric``
  (or by using the :class:`GEval` subclass). Adds chain-of-thought reasoning over
  explicit evaluation steps, a scoring rubric, scale normalization, and a
  configurable ``threshold``. This is the DeepEval-style "G-Eval" judge.

Both paths share the same template renderer and the same tolerant verdict parsing:
a reply that isn't clean JSON is recovered by regex, and an unrecoverable one earns
exactly one retry before the judge reports failure — so a flaky judge reply never
degrades into a silent ``0.0``.
"""

from __future__ import annotations

import json
import re
import warnings
from typing import Any

from fastaiagent._internal.async_utils import run_sync
from fastaiagent.eval.scorer import Scorer, ScorerResult

# A score-band rubric: a list of ``(score_value_on_scale, description)`` anchors,
# e.g. ``[(1, "Mostly wrong"), (3, "Partially correct"), (5, "Fully correct")]``.
Rubric = list[tuple[float, str]]

# Canonical judge-template placeholders are the single-brace {input}/{output}/{expected}.
# {expected_output} and the double-brace forms are legacy aliases: templates authored on
# the control plane (and served to the SDK verbatim by ``Scorer.from_platform``) use all of
# them, so both runtimes interpolate the union rather than rewriting anyone's template.
JUDGE_PLACEHOLDERS = (
    "{{input}}",
    "{{output}}",
    "{{expected}}",
    "{{expected_output}}",
    "{input}",
    "{output}",
    "{expected}",
    "{expected_output}",
)

# Double-brace alternatives come first: ``{{input}}`` contains ``{input}`` as a substring,
# so the reverse order would leave stray braces behind. One pass, so substituted content is
# never rescanned — an ``input`` value containing the literal text "{output}" stays literal.
_PLACEHOLDER_RE = re.compile(
    r"\{\{input\}\}|\{\{output\}\}|\{\{expected_output\}\}|\{\{expected\}\}"
    r"|\{expected_output\}|\{expected\}|\{input\}|\{output\}"
)

_SCORE_RE = re.compile(r'"score"\s*:\s*([\d.]+)')
_REASONING_RE = re.compile(r'"reasoning"\s*:\s*"([^"]+)"')

_RETRY_INSTRUCTION = (
    'Return ONLY the JSON object: {"score": <number>, "reasoning": "<one sentence>"}'
)


def render_judge_template(template: str, input: str, output: str, expected: str) -> str:
    """Interpolate a judge prompt template, accepting every placeholder alias.

    Canonical spelling is ``{input}``/``{output}``/``{expected}``; ``{expected_output}``
    and the double-brace forms are accepted as legacy aliases. Uses a single regex pass
    rather than ``str.format``, so a literal JSON exemplar like ``{"score": ...}`` in the
    template survives untouched.
    """
    mapping = {
        "{{input}}": input,
        "{input}": input,
        "{{output}}": output,
        "{output}": output,
        "{{expected}}": expected,
        "{expected}": expected,
        "{{expected_output}}": expected,
        "{expected_output}": expected,
    }
    return _PLACEHOLDER_RE.sub(lambda m: mapping[m.group(0)], template)


class _UnparseableVerdictError(Exception):
    """The judge replied, but no score could be recovered — even after a retry.

    Distinct from a transport/LLM error so the two report different reasons.
    """

    def __init__(self, raw: str):
        self.raw = raw
        super().__init__(raw)


def _parse_verdict(content: str) -> tuple[float, str] | None:
    """Extract ``(raw_score, reasoning)`` from a judge reply, or ``None`` if unparseable.

    Tries strict JSON first (tolerating markdown fences), then falls back to a regex
    scan — models that wrap the verdict in prose still yield their score instead of
    silently scoring 0.0.
    """
    from fastaiagent.eval.agent_metrics import _strip_fences

    try:
        data = json.loads(_strip_fences(content))
        if isinstance(data, dict) and "score" in data:
            return float(data["score"]), str(data.get("reasoning", ""))
    except Exception:
        pass

    match = _SCORE_RE.search(content)
    if match:
        try:
            raw = float(match.group(1))
        except ValueError:
            return None
        reason_match = _REASONING_RE.search(content)
        return raw, (reason_match.group(1) if reason_match else "Could not parse reasoning")
    return None


async def _complete_and_parse(llm: Any, messages: list[Any]) -> tuple[float, str]:
    """Call the judge and parse its verdict, retrying once on an unparseable reply.

    The retry hands the model its own reply back and demands bare JSON. Transport
    errors propagate untouched; only a genuinely unreadable verdict raises
    :class:`_UnparseableVerdictError`.
    """
    from fastaiagent.llm import AssistantMessage, UserMessage

    response = await llm.acomplete(messages)
    content = response.content or ""
    parsed = _parse_verdict(content)
    if parsed is not None:
        return parsed

    retry = await llm.acomplete(
        [*messages, AssistantMessage(content), UserMessage(_RETRY_INSTRUCTION)]
    )
    retry_content = retry.content or ""
    parsed = _parse_verdict(retry_content)
    if parsed is not None:
        return parsed
    raise _UnparseableVerdictError(retry_content or content)


def _unparseable_result(exc: _UnparseableVerdictError) -> ScorerResult:
    """Failure result for an unreadable verdict — quotes the reply so it's diagnosable."""
    return ScorerResult(
        score=0.0,
        passed=False,
        reason=f"Judge verdict unparseable after retry: {exc.raw[:200]!r}",
    )


def _parse_scale(scale: str) -> tuple[float, float]:
    """Return the ``(low, high)`` bounds for a scale string.

    ``"binary"``/``"0-1"`` → ``(0, 1)``; ``"1-5"`` → ``(1, 5)``; ``"a-b"`` → ``(a, b)``.
    Unknown scales fall back to ``(0, 1)``.
    """
    s = (scale or "0-1").strip().lower()
    if s in ("binary", "0-1", "0_1"):
        return 0.0, 1.0
    if "-" in s:
        lo_str, hi_str = s.split("-", 1)
        try:
            return float(lo_str), float(hi_str)
        except ValueError:
            return 0.0, 1.0
    return 0.0, 1.0


def _normalize_to_unit(raw: float, scale: str) -> float:
    """Map a raw judge score expressed on ``scale`` into the unit interval [0, 1]."""
    lo, hi = _parse_scale(scale)
    if hi == lo:
        return max(0.0, min(1.0, raw))
    return max(0.0, min(1.0, (raw - lo) / (hi - lo)))


def _fmt_num(value: float) -> str:
    """Render whole numbers without a trailing ``.0`` for prompt readability."""
    return str(int(value)) if float(value).is_integer() else str(value)


def _render_rubric(rubric: Rubric) -> str:
    """Render score-band anchors as prompt lines (one ``- Score N: …`` per entry)."""
    return "\n".join(f"- Score {_fmt_num(value)}: {desc}" for value, desc in rubric)


def _build_geval_prompt(
    *,
    criteria: str,
    steps: list[str] | None,
    rubric: Rubric | None,
    scale: str,
    input: str,
    output: str,
    expected: str | None,
    context: str | None,
) -> str:
    """Build the G-Eval chain-of-thought judging prompt.

    Only the fields that are present are included, so ``Expected``/``Context``
    blocks are omitted when not supplied.
    """
    lo, hi = _parse_scale(scale)
    parts: list[str] = [f"You are evaluating an AI response against the criteria: {criteria}."]
    if steps:
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        parts.append("Evaluation steps:\n" + numbered)
    if rubric:
        parts.append("Scoring rubric:\n" + _render_rubric(rubric))
    parts.append(f"Input:\n{input}")
    if expected:
        parts.append(f"Reference / expected output:\n{expected}")
    if context:
        parts.append(f"Context:\n{context}")
    parts.append(f"Actual output:\n{output}")
    parts.append(
        "Work through each evaluation step, then assign a single overall score from "
        f"{_fmt_num(lo)} to {_fmt_num(hi)}.\n"
        'Respond with JSON only: {"score": <number>, "reasoning": "<step-by-step>"}'
    )
    return "\n\n".join(parts)


class LLMJudge(Scorer):
    """Scores output using an LLM as a judge.

    Two modes:

    * **Single-call** (default) — ``LLMJudge(criteria="correctness")``: one LLM call
      against ``prompt_template``.
    * **G-Eval** — pass ``evaluation_steps`` and/or a score-band ``rubric`` (or use
      :class:`GEval`): chain-of-thought over the steps + rubric.

    Either way the raw verdict is normalized from ``scale`` into 0-1 and
    ``passed = score >= threshold``.

    ``prompt_template`` accepts the canonical ``{input}``/``{output}``/``{expected}``
    placeholders as well as the legacy ``{expected_output}`` and double-brace aliases —
    see :func:`render_judge_template`.

    Example:
        judge = LLMJudge(criteria="correctness")
        result = judge.score(input="What is 2+2?", output="4", expected="4")

        geval = LLMJudge(
            criteria="correctness",
            evaluation_steps=["Compare each claim to the expected answer."],
            rubric=[(1, "Wrong"), (5, "Fully correct")],
            scale="1-5",
        )
    """

    name = "llm_judge"

    def __init__(
        self,
        criteria: str = "correctness",
        prompt_template: str | None = None,
        llm: Any = None,
        scale: str = "binary",  # "binary", "0-1", "1-5", "1-10", ...
        *,
        evaluation_steps: list[str] | None = None,
        rubric: Rubric | None = None,
        threshold: float = 0.5,
        name: str | None = None,
        auto_steps: bool = False,
    ):
        self.criteria = criteria
        self.prompt_template = prompt_template or self._default_prompt()
        self._llm = llm
        self.scale = scale
        self.evaluation_steps = evaluation_steps
        self.rubric = rubric
        self.threshold = threshold
        self.auto_steps = auto_steps
        self._force_geval = False
        if name is not None:
            self.name = name
        if prompt_template and not any(p in prompt_template for p in JUDGE_PLACEHOLDERS):
            # Non-blocking lint, mirroring the control plane: a template with no
            # placeholder means the judge never sees the content it is scoring.
            warnings.warn(
                "Judge prompt_template contains no known placeholder "
                "({input}, {output}, {expected}); the judge will not see the "
                "evaluated content.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def _geval_mode(self) -> bool:
        """G-Eval activates when steps/rubric are supplied, or a subclass forces it."""
        return self.evaluation_steps is not None or self.rubric is not None or self._force_geval

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
        """Score using LLM judge (async) — dispatches legacy vs. G-Eval."""
        if not self._geval_mode:
            return await self._ascore_legacy(input, output, expected, **kwargs)
        return await self._ascore_geval(input, output, expected, **kwargs)

    async def _ascore_legacy(
        self, input: str, output: str, expected: str | None = None, **kwargs: Any
    ) -> ScorerResult:
        """Single-call judge — one LLM call against ``prompt_template``."""
        from fastaiagent.llm import LLMClient, SystemMessage, UserMessage

        llm = self._llm or LLMClient()

        prompt = render_judge_template(self.prompt_template, input, output, expected or "N/A")

        try:
            raw, reasoning = await _complete_and_parse(
                llm,
                [
                    SystemMessage("You are an evaluation judge. Respond with JSON only."),
                    UserMessage(prompt),
                ],
            )
        except _UnparseableVerdictError as e:
            return _unparseable_result(e)
        except Exception as e:
            return ScorerResult(score=0.0, passed=False, reason=f"Judge error: {e}")

        score_val = _normalize_to_unit(raw, self.scale)
        return ScorerResult(
            score=score_val, passed=score_val >= self.threshold, reason=reasoning
        )

    async def _ascore_geval(
        self, input: str, output: str, expected: str | None = None, **kwargs: Any
    ) -> ScorerResult:
        """G-Eval path — chain-of-thought over evaluation steps + score-band rubric."""
        from fastaiagent.eval.agent_metrics import _resolve_context
        from fastaiagent.llm import LLMClient, SystemMessage, UserMessage

        llm = self._llm or LLMClient()
        context = _resolve_context(kwargs)

        try:
            steps = await self._ensure_steps(llm)
            prompt = _build_geval_prompt(
                criteria=self.criteria,
                steps=steps,
                rubric=self.rubric,
                scale=self.scale,
                input=input,
                output=output,
                expected=expected,
                context=context,
            )
            raw, reasoning = await _complete_and_parse(
                llm,
                [
                    SystemMessage(
                        "You are a meticulous evaluation judge. Reason step by step, "
                        "then respond with JSON only."
                    ),
                    UserMessage(prompt),
                ],
            )
        except _UnparseableVerdictError as e:
            return _unparseable_result(e)
        except Exception as e:
            return ScorerResult(score=0.0, passed=False, reason=f"Judge error: {e}")

        score_val = _normalize_to_unit(raw, self.scale)
        return ScorerResult(
            score=score_val,
            passed=score_val >= self.threshold,
            reason=reasoning,
        )

    async def _ensure_steps(self, llm: Any) -> list[str] | None:
        """Lazily derive evaluation steps from ``criteria`` (G-Eval "Auto-CoT").

        Runs only when ``auto_steps`` is set and no explicit steps were given; the
        result is cached on the instance so it costs at most one extra LLM call.
        Best-effort: on any error it falls back to a criteria-only prompt.
        """
        if self.evaluation_steps is not None or not self.auto_steps:
            return self.evaluation_steps

        from fastaiagent.eval.agent_metrics import _strip_fences
        from fastaiagent.llm import SystemMessage, UserMessage

        prompt = (
            f"Given the evaluation criteria: '{self.criteria}', generate 3-4 concise, "
            "ordered steps an expert judge should follow to evaluate a response.\n"
            'Respond with JSON only: {"steps": ["step 1", "step 2", ...]}'
        )
        try:
            response = await llm.acomplete(
                [
                    SystemMessage("You design evaluation rubrics. Respond with JSON only."),
                    UserMessage(prompt),
                ]
            )
            data = json.loads(_strip_fences(response.content or "{}"))
            steps = [str(s) for s in data.get("steps", []) if str(s).strip()]
            self.evaluation_steps = steps or None
        except Exception:
            self.evaluation_steps = None
        return self.evaluation_steps


class GEval(LLMJudge):
    """G-Eval judge — criteria + evaluation steps + score-band rubric + chain-of-thought.

    A DeepEval-familiar constructor over :class:`LLMJudge`'s G-Eval mode. Defaults to a
    ``1-5`` scale and auto-generates evaluation steps from ``criteria`` when none are
    supplied. The final score is normalized to 0-1, so ``threshold`` is on the 0-1 scale
    (``0.5`` ≈ the middle of the rubric).

    Example:
        judge = GEval(
            name="correctness",
            criteria="Is the answer factually correct and complete?",
            evaluation_steps=[
                "Check each claim in the output against the expected answer.",
                "Penalize fabricated or contradicted facts.",
            ],
            rubric=[(1, "Mostly incorrect"), (3, "Partially correct"), (5, "Fully correct")],
            scale="1-5",
        )
        result = judge.score(input="Capital of France?", output="Paris", expected="Paris")
    """

    def __init__(
        self,
        name: str = "g_eval",
        criteria: str = "correctness",
        evaluation_steps: list[str] | None = None,
        rubric: Rubric | None = None,
        scale: str = "1-5",
        threshold: float = 0.5,
        llm: Any = None,
        auto_steps: bool = True,
    ):
        super().__init__(
            criteria=criteria,
            prompt_template=None,
            llm=llm,
            scale=scale,
            evaluation_steps=evaluation_steps,
            rubric=rubric,
            threshold=threshold,
            name=name,
            auto_steps=auto_steps,
        )
        self._force_geval = True
