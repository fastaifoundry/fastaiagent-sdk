"""The ``backend="decisions"`` engine for model-backed guardrails (1.84.0).

``llm_judge``, ``content_safety`` and ``topic`` ask a chat model for JSON and parse
a verdict out of free text. With ``config["backend"] = "decisions"`` they ask
OpenAI's Decisions API instead: the payload goes in as **evidence** and each check
is a :class:`~fastaiagent.llm.decisions.Predicate`, so the reply is a probability
per question — no prompt to inject through and nothing to parse.

``backend`` is the same key ``pii`` already uses (``regex`` | ``presidio``): *which
engine runs this rule*. Default ``"chat"`` keeps every existing rule unchanged.

This module is the SDK's. The plane-canonical ``topics.py``,
``hazard_taxonomy.py`` and ``grounding.py`` are read here (category labels, topic
resolution, polarity) and never edited, so their mirror stays byte-faithful.

Everything that cannot give a verdict **raises**, and ``run_guardrail`` turns that
into ``errored=True`` under the rule's ``on_error`` (CLAUDE.md §2.4):

* an unknown ``backend`` value — a typo must not silently fall back to ``chat``;
* ``backend="decisions"`` on a type it does not cover (``groundedness`` would lose
  ``unsupported_claims``, its evidence);
* a refusal on any question — the rule did not get its answer;
* ``llm_judge`` without ``instructions`` (the condition PASS requires).
"""

from __future__ import annotations

from typing import Any

from fastaiagent.llm.decisions import DEFAULT_DECISION_MODEL, DecisionResult, Predicate

BACKENDS = ("chat", "decisions")

#: Types that run on the Decisions API. Anything else given
#: ``backend="decisions"`` raises rather than quietly using chat.
DECISION_TYPES = ("llm_judge", "content_safety", "topic")


def resolve_backend(config: dict[str, Any], type_name: str) -> str:
    """``"chat"`` (default) or ``"decisions"``; anything else raises."""
    raw = config.get("backend")
    backend = str(raw).strip().lower() if raw not in (None, "") else "chat"
    if backend not in BACKENDS:
        raise ValueError(
            f"{type_name} guardrail backend {raw!r} is not one of {', '.join(BACKENDS)}. "
            f"Refusing to guess which engine was meant."
        )
    if backend == "decisions" and type_name not in DECISION_TYPES:
        raise ValueError(
            f"backend='decisions' is not supported for {type_name} guardrails "
            f"(supported: {', '.join(DECISION_TYPES)})."
        )
    return backend


def decision_client(config: dict[str, Any]) -> Any:
    """The client a decisions-backed rule calls.

    ``config["llm"]`` (``LLMClient`` kwargs) when set, with ``model`` defaulting to
    the Decisions model rather than ``LLMClient``'s chat default; otherwise the
    Decisions model with the detector retry budget.
    """
    from fastaiagent._internal.safety_detectors import LLM_DETECTOR_MAX_RETRIES
    from fastaiagent.llm import LLMClient

    llm_config = dict(config.get("llm") or {})
    if llm_config:
        llm_config.setdefault("model", DEFAULT_DECISION_MODEL)
        return LLMClient(**llm_config)
    return LLMClient(model=DEFAULT_DECISION_MODEL, max_retries=LLM_DETECTOR_MAX_RETRIES)


def _require_answers(result: DecisionResult, type_name: str) -> None:
    if result.refused:
        names = [r.name for r in result.refusals]
        raise ValueError(
            f"{type_name} guardrail: the Decisions API refused {names}; the rule could not "
            f"run, so it reports no verdict."
        )


def _threshold(config: dict[str, Any], key: str, default: float = 0.5) -> float:
    raw = config.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{key} must be a number in 0..1, got {raw!r}") from e
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be in 0..1, got {value}")
    return value


async def judge_probability(config: dict[str, Any], text: str) -> tuple[float, float, str]:
    """``llm_judge``: probability that ``config["instructions"]`` holds.

    Returns ``(probability, threshold, instructions)``.
    """
    instructions = config.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError(
            "llm_judge with backend='decisions' needs config['instructions']: the "
            "condition the content must meet to PASS (a statement, not a PASS/FAIL "
            "prompt). Without it there is nothing to decide."
        )
    threshold = _threshold(config, "threshold")
    result = await decision_client(config).adecide(
        text, [Predicate(name="pass", instructions=instructions)]
    )
    _require_answers(result, "llm_judge")
    return result.predicates["pass"].probability, threshold, instructions


async def hazard_scores(config: dict[str, Any], categories: list[str], text: str) -> dict[str, float]:
    """``content_safety``: one predicate per hazard category → probability as its score."""
    from fastaiagent.guardrail.hazard_taxonomy import MLCOMMONS_HAZARDS

    questions = [
        Predicate(
            name=code,
            instructions=(
                f"The content contains or promotes {MLCOMMONS_HAZARDS[code][0].lower()} "
                f"({MLCOMMONS_HAZARDS[code][1]})."
            ),
        )
        for code in categories
    ]
    result = await decision_client(config).adecide(text, questions)
    _require_answers(result, "content_safety")
    return {code: result.predicates[code].probability for code in categories}


async def topics_present(
    config: dict[str, Any], topics: list[dict[str, str]], text: str
) -> list[str]:
    """``topic``: names whose predicate clears ``topic_threshold`` (default 0.5).

    One predicate per topic, because a payload can be about several at once — a
    single-answer ``Choice`` would hide every topic but one.
    """
    threshold = _threshold(config, "topic_threshold")
    questions = [
        Predicate(
            name=f"t{i}",
            instructions=(
                f"The content is about {t['name']}"
                + (f": {t['description']}" if t.get("description") else "")
                + "."
            ),
        )
        for i, t in enumerate(topics)
    ]
    result = await decision_client(config).adecide(text, questions)
    _require_answers(result, "topic")
    return [
        t["name"]
        for i, t in enumerate(topics)
        if result.predicates[f"t{i}"].probability >= threshold
    ]


__all__ = [
    "BACKENDS",
    "DECISION_TYPES",
    "decision_client",
    "hazard_scores",
    "judge_probability",
    "resolve_backend",
    "topics_present",
]
