"""Decision-routed condition nodes (1.84.0).

A ``condition`` node normally routes on ``conditions=[{"expression", "handle"}]``
— string comparisons over state. With ``decision=`` it routes on what the
content *means*: OpenAI's Decisions API answers a
:class:`~fastaiagent.llm.decisions.Choice` about the rendered input, and the
chosen value is the handle, so edges are labelled with option values::

    chain.add_node(
        "triage",
        type=NodeType.condition,
        decision={
            "question": Choice(instructions="Which team?", options=["billing", "technical"]),
            "input": "{{input.message}}",         # default "{{input}}"
            "llm": {"model": "gpt-6-luna"},       # LLMClient kwargs (default model)
            "min_confidence": 0.6,                # below it → "default"
        },
    )
    chain.connect("triage", "billing_agent", label="billing")
    chain.connect("triage", "tech_agent", label="technical")
    chain.connect("triage", "human")              # default: refusal / low confidence

No new node type: the edge-selection code and the topology every consumer reads
are unchanged. The config is stored as plain JSON (the question in its wire form,
the client as kwargs) so a chain still round-trips through ``to_dict`` — which is
also why ``llm`` takes kwargs, not a live client.

Boolean options route on ``"True"`` / ``"False"`` labels.
"""

from __future__ import annotations

from typing import Any

from fastaiagent.llm.decisions import (
    DEFAULT_DECISION_MODEL,
    Choice,
    ChoiceAnswer,
    question_from_dict,
)


def normalize_decision_config(raw: Any, node_id: str) -> dict[str, Any]:
    """Validate ``add_node(decision=...)`` and store it as JSON.

    Raises ``ValueError`` / ``TypeError`` at build time for anything that could not
    route: a non-``Choice`` question, a live client (not serialisable), a
    ``min_confidence`` outside 0..1.
    """
    if isinstance(raw, Choice):
        raw = {"question": raw}
    if not isinstance(raw, dict) or "question" not in raw:
        raise ValueError(
            f"Condition node {node_id!r}: decision= needs a Choice, or a dict with a "
            f"'question' Choice (plus optional 'input', 'llm', 'min_confidence')."
        )
    unknown = sorted(set(raw) - {"question", "input", "llm", "min_confidence"})
    if unknown:
        raise ValueError(f"Condition node {node_id!r}: unknown decision key(s) {unknown}")

    question = raw["question"]
    if isinstance(question, dict):
        question = question_from_dict(question)
    if not isinstance(question, Choice):
        raise ValueError(
            f"Condition node {node_id!r}: a decision routes on a Choice (one value per "
            f"outgoing edge); got {type(question).__name__}. For yes/no use "
            f"Choice(options=[True, False]) and label the edges 'True' / 'False'."
        )

    llm = raw.get("llm") or {}
    if not isinstance(llm, dict):
        raise TypeError(
            f"Condition node {node_id!r}: decision 'llm' takes LLMClient kwargs "
            f"(e.g. {{'model': 'gpt-6-luna'}}), not a client object — node config is "
            f"serialised with the chain."
        )

    min_conf = raw.get("min_confidence", 0.0)
    try:
        min_conf = float(min_conf)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Condition node {node_id!r}: min_confidence must be a number") from e
    if not 0.0 <= min_conf <= 1.0:
        raise ValueError(f"Condition node {node_id!r}: min_confidence must be in 0..1")

    template = raw.get("input", "{{input}}")
    if not isinstance(template, str) or not template.strip():
        raise ValueError(f"Condition node {node_id!r}: decision 'input' must be a template string")

    return {
        "question": question.to_wire(),
        "input": template,
        "llm": dict(llm),
        "min_confidence": min_conf,
    }


def decision_handles(decision: dict[str, Any]) -> set[str]:
    """Every handle a decision node can return — one per option, plus ``default``."""
    q = question_from_dict(decision["question"])
    assert isinstance(q, Choice)
    return {str(o.value) for o in q.options} | {"default"}


async def run_decision_condition(
    decision: dict[str, Any], context: dict[str, Any], render: Any
) -> dict[str, Any]:
    """Ask the node's Choice about the rendered input; return ``{"matched": handle, ...}``.

    A refusal, or a confidence below ``min_confidence``, routes ``"default"`` —
    the branch a chain reserves for "could not decide", never a guessed option.
    A failed call raises, failing the node like any other node error.
    """
    from fastaiagent.llm import LLMClient

    question = question_from_dict(decision["question"])
    evidence = render(decision.get("input", "{{input}}"), context)
    llm = LLMClient(**{"model": DEFAULT_DECISION_MODEL, **(decision.get("llm") or {})})
    result = await llm.adecide(evidence, [question])
    answer = result[0]
    min_conf = float(decision.get("min_confidence", 0.0))

    if isinstance(answer, ChoiceAnswer) and answer.confidence >= min_conf:
        handle = str(answer.choice)
    else:
        handle = "default"
    return {
        "matched": handle,
        "decision": {
            "choice": answer.choice if isinstance(answer, ChoiceAnswer) else None,
            "confidence": answer.confidence if isinstance(answer, ChoiceAnswer) else None,
            "refused": not isinstance(answer, ChoiceAnswer),
        },
    }


__all__ = ["decision_handles", "normalize_decision_config", "run_decision_condition"]
