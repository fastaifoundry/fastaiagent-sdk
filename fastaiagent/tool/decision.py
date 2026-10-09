"""``decision_tool`` — give an agent the Decisions API as a tool (1.84.0).

The agent passes the text it wants classified; the tool asks fixed questions about
it and returns the answers as data. Useful when *the agent* should decide when to
classify — triage, routing inside a Swarm or Supervisor, "is this request in
scope?" — at a fraction of a chat turn's latency and price::

    triage = decision_tool(
        {"department": Choice(instructions="Which team?", options=["billing", "technical", "other"]),
         "urgent": Predicate(instructions="The customer needs help right now.")},
        llm=LLMClient(model="gpt-6-luna"),
        name="triage_ticket",
    )
    agent = Agent(name="support", llm=chat_llm, tools=[triage])

The return value is a dict keyed by question name: ``{"department": {"choice":
"billing", "confidence": 0.93, ...}, "urgent": {"probability": 0.81}}``. A refused
question comes back as ``{"refused": true}`` so the agent sees it was not answered.
"""

from __future__ import annotations

from typing import Any

from fastaiagent.llm.decisions import (
    DEFAULT_DECISION_MODEL,
    ChoiceAnswer,
    PredicateAnswer,
    ScoreAnswer,
    normalize_questions,
)
from fastaiagent.tool.function import FunctionTool


def _answer_dict(answer: Any) -> dict[str, Any]:
    if isinstance(answer, PredicateAnswer):
        return {"probability": answer.probability}
    if isinstance(answer, ChoiceAnswer):
        return {
            "choice": answer.choice,
            "confidence": answer.confidence,
            "probabilities": {str(p.value): p.probability for p in answer.probabilities},
        }
    if isinstance(answer, ScoreAnswer):
        return {
            "score": answer.score,
            "normalized": answer.normalized,
            "level": answer.level,
            "confidence": answer.confidence,
        }
    return {"refused": True}


def decision_tool(
    questions: Any,
    *,
    llm: Any = None,
    name: str = "decide",
    description: str | None = None,
) -> FunctionTool:
    """Build a :class:`~fastaiagent.tool.FunctionTool` that asks ``questions``.

    Args:
        questions: One ``Predicate`` / ``Choice`` / ``Score``, a list, or a
            ``{name: question}`` mapping. Every question needs a name (the key
            the agent reads the answer under); a single unnamed question is
            named ``"answer"``.
        llm: An ``LLMClient`` (or anything with ``adecide``, e.g. ``TestModel``).
            Defaults to ``LLMClient(model="gpt-6-luna")``.
        name: Tool name the model calls.
        description: Tool description; generated from the questions if omitted.
    """
    qs = normalize_questions(questions)
    if len(qs) == 1 and qs[0].name is None:
        qs = [qs[0].model_copy(update={"name": "answer"})]
    unnamed = [i for i, q in enumerate(qs) if q.name is None]
    if unnamed:
        raise ValueError(
            "decision_tool questions must be named (the agent reads each answer by "
            "name); pass a {name: question} mapping or set name= on each."
        )

    if description is None:
        asked = "; ".join(f"{q.name}: {q.instructions}" for q in qs)
        description = (
            "Classify a piece of text with fixed-answer questions and return the "
            f"answers with their probabilities. Questions — {asked}"
        )

    client = llm

    async def _decide(input: str) -> dict[str, Any]:
        nonlocal client
        if client is None:
            from fastaiagent.llm import LLMClient

            client = LLMClient(model=DEFAULT_DECISION_MODEL)
        result = await client.adecide(input, qs)
        # Every question is named (checked above), so ``str(q.name)`` is exact.
        return {str(q.name): _answer_dict(a) for q, a in zip(qs, result.answers, strict=True)}

    return FunctionTool(
        name=name,
        fn=_decide,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "The text to classify."}
            },
            "required": ["input"],
        },
    )


__all__ = ["decision_tool"]
