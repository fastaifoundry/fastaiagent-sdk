"""OpenAI Decisions API — typed questions in, typed answers out.

The Decisions API (``POST /v1/decisions``, public beta since 2026-10-06) is not a
chat endpoint. You hand it shared evidence (text and/or inline images) and a list
of questions whose possible answers you fixed in advance; it returns, per
question, a probability distribution rather than generated text:

=============  =========================================  ==============================
Question       You supply                                 Main result
=============  =========================================  ==============================
``Predicate``  a condition (``instructions``)             ``probability`` (0..1) it holds
``Choice``     2+ options (``str`` or ``bool`` values)     ``choice`` + per-option probabilities
``Score``      2+ ordered levels                          ``score`` = Σ p·index (0..n-1)
=============  =========================================  ==============================

Any single answer may come back as a :class:`Refusal` while the others in the same
request are still answered.

This module holds the request/response types and the input normalisation. The
call itself is :meth:`fastaiagent.llm.LLMClient.adecide` /
:meth:`~fastaiagent.llm.LLMClient.decide`, so the API key, ``base_url``, TLS
``verify``, ``max_retries`` and an injected ``openai_client`` are all the ones the
client already carries.

Every shape here follows OpenAI's own published types (``openai`` 3.26.0,
``types/decision*.py``) and was checked against the live endpoint.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Annotated, Any, Literal, Union

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from fastaiagent._internal.errors import LLMError

#: A choice value. Typed on the wire: the string ``"true"`` and the boolean
#: ``True`` are different options, so neither is coerced into the other.
ChoiceValue = Union[StrictStr, StrictBool]  # noqa: UP007 — Annotated aliases, runtime union

#: The only model the endpoint serves today (OpenAI Decisions guide).
DEFAULT_DECISION_MODEL = "gpt-6-luna"


def _require_text(value: str, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")
    return value


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------


class Option(BaseModel):
    """One allowed answer of a :class:`Choice`."""

    model_config = ConfigDict(frozen=True)

    value: ChoiceValue
    description: str | None = None

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"value": self.value}
        if self.description:
            out["description"] = self.description
        return out


class Level(BaseModel):
    """One ordered level of a :class:`Score`. Index 0 is the lowest."""

    model_config = ConfigDict(frozen=True)

    label: str
    description: str | None = None

    @field_validator("label")
    @classmethod
    def _label(cls, v: str) -> str:
        return _require_text(v, "Level label")

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"label": self.label}
        if self.description:
            out["description"] = self.description
        return out


class _QuestionBase(BaseModel):
    model_config = ConfigDict(frozen=True)

    instructions: str
    name: str | None = None

    @field_validator("instructions")
    @classmethod
    def _instructions(cls, v: str) -> str:
        # A question with nothing to ask cannot decide anything. Refusing it here
        # is what keeps an empty rule from ever coming back as a clean answer.
        return _require_text(v, f"{cls.__name__} instructions")

    def _wire_base(self, type_: str) -> dict[str, Any]:
        out: dict[str, Any] = {"type": type_, "instructions": self.instructions}
        if self.name is not None:
            out["name"] = self.name
        return out


class Predicate(_QuestionBase):
    """Estimate how likely a condition about the input is to be true.

    Answered by :class:`PredicateAnswer` (``probability`` in 0..1).

    Example::

        Predicate(name="damaged",
                  instructions="The product shows visible damage such as a crack or dent.")
    """

    type: Literal["predicate"] = "predicate"

    def to_wire(self) -> dict[str, Any]:
        return self._wire_base("predicate")


class Choice(_QuestionBase):
    """Pick exactly one of a fixed set of unordered options.

    ``options`` accepts a list of values (``["billing", "technical"]``), a mapping
    of value to description (``{"billing": "Payments and refunds", ...}``), or a
    list of :class:`Option`. ``choices`` (the wire name) is accepted as an alias.
    Values may be ``str`` or ``bool`` and are kept typed.

    Answered by :class:`ChoiceAnswer`.
    """

    type: Literal["choice"] = "choice"
    options: list[Option] = Field(validation_alias=AliasChoices("options", "choices"))

    @field_validator("options", mode="before")
    @classmethod
    def _coerce_options(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return [{"value": k, "description": d} for k, d in v.items()]
        if isinstance(v, (str, bytes)) or not isinstance(v, Iterable):
            raise ValueError("Choice options must be a list of values or a {value: description} map")
        out: list[Any] = []
        for item in v:
            if isinstance(item, (str, bool)):
                out.append({"value": item})
            else:
                out.append(item)
        return out

    @field_validator("options")
    @classmethod
    def _check_options(cls, v: list[Option]) -> list[Option]:
        if len(v) < 2:
            # One option can only ever be chosen; it decides nothing.
            raise ValueError("Choice needs at least 2 options")
        seen: set[tuple[type, Any]] = set()
        for opt in v:
            key = (type(opt.value), opt.value)
            if key in seen:
                raise ValueError(f"Choice option {opt.value!r} is listed twice")
            seen.add(key)
        return v

    def to_wire(self) -> dict[str, Any]:
        out = self._wire_base("choice")
        out["choices"] = [o.to_wire() for o in self.options]
        return out


class Score(_QuestionBase):
    """Rate the input on an ordered scale of 2+ levels (lowest first).

    ``levels`` accepts a list of labels or of :class:`Level`. The answer's
    ``score`` is the probability-weighted level index — between ``0`` and
    ``len(levels) - 1``, and it can fall between levels.
    :attr:`ScoreAnswer.normalized` rescales it to 0..1.

    Answered by :class:`ScoreAnswer`.
    """

    type: Literal["score"] = "score"
    levels: list[Level]

    @field_validator("levels", mode="before")
    @classmethod
    def _coerce_levels(cls, v: Any) -> Any:
        if isinstance(v, (str, bytes)) or not isinstance(v, Iterable):
            raise ValueError("Score levels must be a list of labels or Level objects")
        return [{"label": item} if isinstance(item, str) else item for item in v]

    @field_validator("levels")
    @classmethod
    def _check_levels(cls, v: list[Level]) -> list[Level]:
        if len(v) < 2:
            raise ValueError("Score needs at least 2 levels")
        labels = [lvl.label for lvl in v]
        if len(set(labels)) != len(labels):
            raise ValueError("Score level labels must be unique")
        return v

    def to_wire(self) -> dict[str, Any]:
        out = self._wire_base("score")
        out["levels"] = [lvl.to_wire() for lvl in self.levels]
        return out


Question = Annotated[Predicate | Choice | Score, Field(discriminator="type")]
_QUESTION_ADAPTER: TypeAdapter[Predicate | Choice | Score] = TypeAdapter(Question)


def question_from_dict(data: dict[str, Any]) -> Predicate | Choice | Score:
    """Build a question from its dict form (wire shape or ``model_dump()``)."""
    return _QUESTION_ADAPTER.validate_python(data)


def normalize_questions(
    questions: Predicate | Choice | Score | dict[str, Any] | Sequence[Any],
) -> list[Predicate | Choice | Score]:
    """Coerce questions into a validated list.

    Accepts one question, a list (objects or their dict form), or a mapping of
    ``{name: question}`` — the key becomes the question's ``name``.

    Raises ``ValueError`` on an empty list or a repeated name: answers are looked
    up by name, so two questions sharing one would make a result ambiguous.
    """
    if isinstance(questions, dict) and "type" not in questions:
        items: list[Any] = []
        for key, q in questions.items():
            if isinstance(q, dict):
                q = question_from_dict(q)
            if not isinstance(q, (Predicate, Choice, Score)):
                raise TypeError(
                    f"questions[{key!r}] must be a Predicate / Choice / Score; "
                    f"got {type(q).__name__}"
                )
            if q.name is not None and q.name != key:
                raise ValueError(f"questions[{key!r}] is already named {q.name!r}")
            items.append(q.model_copy(update={"name": key}))
    elif isinstance(questions, (Predicate, Choice, Score, dict)):
        items = [questions]
    else:
        items = list(questions)
    if not items:
        raise ValueError("decide() needs at least one question")
    out: list[Predicate | Choice | Score] = []
    for q in items:
        if isinstance(q, (Predicate, Choice, Score)):
            out.append(q)
        elif isinstance(q, dict):
            out.append(question_from_dict(q))
        else:
            raise TypeError(
                f"questions must be Predicate / Choice / Score (or their dict form); "
                f"got {type(q).__name__}"
            )
    names = [q.name for q in out if q.name is not None]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"question names must be unique within a request; repeated: {dupes}")
    return out


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


class PredicateAnswer(BaseModel):
    """``probability`` (0..1) that the predicate's condition holds."""

    type: Literal["predicate"] = "predicate"
    name: str | None = None
    probability: float


class ChoiceProbability(BaseModel):
    value: ChoiceValue
    probability: float


class ChoiceAnswer(BaseModel):
    """The chosen option, every option's probability, and a ``confidence``."""

    type: Literal["choice"] = "choice"
    name: str | None = None
    choice: ChoiceValue
    probabilities: list[ChoiceProbability] = Field(default_factory=list)
    confidence: float

    def probability_of(self, value: ChoiceValue) -> float:
        """Probability of one option (typed match: ``"true"`` is not ``True``)."""
        for p in self.probabilities:
            if type(p.value) is type(value) and p.value == value:
                return p.probability
        raise KeyError(value)


class LevelProbability(BaseModel):
    value: int
    label: str
    probability: float


class ScoreAnswer(BaseModel):
    """``score`` is Σ probability × level index, so it runs 0..n-1, not 0..1."""

    type: Literal["score"] = "score"
    name: str | None = None
    score: float
    probabilities: list[LevelProbability] = Field(default_factory=list)
    confidence: float

    @property
    def normalized(self) -> float:
        """``score`` rescaled to 0..1 (``score / (levels - 1)``)."""
        top = len(self.probabilities) - 1
        return self.score / top if top > 0 else 0.0

    @property
    def level(self) -> str | None:
        """Label of the single most likely level."""
        if not self.probabilities:
            return None
        return max(self.probabilities, key=lambda p: p.probability).label


class Refusal(BaseModel):
    """The model declined this one question. It carries no score by design."""

    type: Literal["refusal"] = "refusal"
    name: str | None = None


Answer = Annotated[
    PredicateAnswer | ChoiceAnswer | ScoreAnswer | Refusal, Field(discriminator="type")
]
_ANSWERS_ADAPTER: TypeAdapter[list[PredicateAnswer | ChoiceAnswer | ScoreAnswer | Refusal]] = (
    TypeAdapter(list[Answer])
)


class DecisionResult(BaseModel):
    """What one ``decide()`` call returned.

    ``answers`` are in question order. Index by position (``result[0]``) or by
    question name (``result["department"]``).
    """

    answers: list[Answer]
    model: str = ""
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
    #: USD for this call, or ``None`` when the model has no known decision rate.
    cost_usd: float | None = None
    #: OpenAI's ``x-request-id`` — quote it when raising an issue with OpenAI.
    request_id: str | None = None

    def __getitem__(self, key: int | str) -> PredicateAnswer | ChoiceAnswer | ScoreAnswer | Refusal:
        if isinstance(key, int):
            return self.answers[key]
        for a in self.answers:
            if a.name == key:
                return a
        raise KeyError(key)

    def __len__(self) -> int:
        return len(self.answers)

    def get(
        self, name: str, default: Any = None
    ) -> PredicateAnswer | ChoiceAnswer | ScoreAnswer | Refusal | Any:
        try:
            return self[name]
        except KeyError:
            return default

    @property
    def predicates(self) -> dict[str, PredicateAnswer]:
        """Predicate answers keyed by question name (unnamed ones omitted)."""
        return {a.name: a for a in self.answers if isinstance(a, PredicateAnswer) and a.name}

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        """Choice answers keyed by question name (unnamed ones omitted)."""
        return {a.name: a for a in self.answers if isinstance(a, ChoiceAnswer) and a.name}

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        """Score answers keyed by question name (unnamed ones omitted)."""
        return {a.name: a for a in self.answers if isinstance(a, ScoreAnswer) and a.name}

    @property
    def refusals(self) -> list[Refusal]:
        return [a for a in self.answers if isinstance(a, Refusal)]

    @property
    def refused(self) -> bool:
        """True when at least one question was refused."""
        return bool(self.refusals)


def parse_decision(
    data: dict[str, Any], questions: Sequence[Predicate | Choice | Score]
) -> DecisionResult:
    """Turn the endpoint's JSON into a :class:`DecisionResult`.

    Raises :class:`LLMError` — never guesses — when the response is not readable
    or does not answer each question exactly once in order. An unknown answer
    ``type`` (the API is in beta) is named in the error rather than dropped.
    """
    raw = data.get("answers")
    if not isinstance(raw, list):
        raise LLMError(f"Decisions API response has no 'answers' list: {str(data)[:300]}")
    try:
        answers = _ANSWERS_ADAPTER.validate_python(raw)
    except ValidationError as e:
        types = sorted({str(a.get("type")) for a in raw if isinstance(a, dict)})
        raise LLMError(
            f"Decisions API returned answers this SDK cannot read (types {types}): {e}"
        ) from e
    if len(answers) != len(questions):
        raise LLMError(
            f"Decisions API answered {len(answers)} question(s) but {len(questions)} were "
            f"asked; answers cannot be matched to questions."
        )
    for q, a in zip(questions, answers, strict=True):
        if isinstance(a, Refusal):
            continue
        if a.type != q.type:
            raise LLMError(
                f"Decisions API answered question {q.name!r} ({q.type}) with a "
                f"{a.type} answer."
            )
    return DecisionResult(
        answers=answers,
        model=str(data.get("model") or ""),
        usage=dict(data.get("usage") or {}),
    )


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

_UNSUPPORTED_INPUT = (
    "The Decisions API accepts only text and inline images (user messages); "
    "PDFs, files, audio, and system/assistant/tool messages are not supported."
)


def _image_part(image: Any, image_cap_mb: float) -> dict[str, Any]:
    from fastaiagent.multimodal.resize import maybe_resize

    img = maybe_resize(image, max_mb=image_cap_mb)
    part: dict[str, Any] = {
        "type": "input_image",
        # Inline data URL only: OpenAI's Decisions types reject external URLs
        # and file ids, so an Image built with ``from_url`` is sent as its bytes.
        "image_url": f"data:{img.media_type};base64,{img.to_base64()}",
    }
    if img.detail and img.detail != "auto":
        part["detail"] = img.detail
    return part


def _parts(content: Any, image_cap_mb: float) -> list[dict[str, Any]]:
    from fastaiagent.multimodal.image import Image

    items = content if isinstance(content, list) else [content]
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            out.append({"type": "input_text", "text": item})
        elif isinstance(item, Image):
            out.append(_image_part(item, image_cap_mb))
        else:
            raise LLMError(f"{_UNSUPPORTED_INPUT} Got {type(item).__name__}.")
    return out


def build_decision_input(input: Any, *, image_cap_mb: float) -> str | list[dict[str, Any]]:
    """Normalise ``decide()``'s ``input`` into the endpoint's ``input`` field.

    Accepts a ``str``; an ``Image``; a list mixing ``str`` and ``Image`` (one user
    message); or a list of user :class:`~fastaiagent.llm.Message` objects.
    """
    from fastaiagent.llm.message import Message, MessageRole

    if isinstance(input, str):
        return input
    if isinstance(input, Message):
        input = [input]
    if isinstance(input, list) and input and all(isinstance(m, Message) for m in input):
        messages: list[dict[str, Any]] = []
        for m in input:
            if m.role != MessageRole.user:
                raise LLMError(f"{_UNSUPPORTED_INPUT} Got a {m.role.value} message.")
            content = m.content if m.content is not None else ""
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": _parts(content, image_cap_mb)})
        return messages
    if isinstance(input, list) and any(isinstance(m, Message) for m in input):
        raise LLMError("decide() input mixes Message objects with raw parts; pass one or the other.")
    return [{"role": "user", "content": _parts(input, image_cap_mb)}]


def summarize_decision_input(input: Any) -> str | None:
    """A render-free JSON summary of ``input`` for the span (never image bytes)."""
    from fastaiagent.llm.message import Message, _summarize_parts

    try:
        if isinstance(input, str):
            return json.dumps(input)
        if isinstance(input, Message):
            input = [input]
        if isinstance(input, list) and input and all(isinstance(m, Message) for m in input):
            return json.dumps([m.to_openai_format() for m in input], default=str)
        items = input if isinstance(input, list) else [input]
        return json.dumps(_summarize_parts(items), default=str)
    except Exception:
        return None


__all__ = [
    "DEFAULT_DECISION_MODEL",
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "ChoiceProbability",
    "ChoiceValue",
    "DecisionResult",
    "Level",
    "LevelProbability",
    "Option",
    "Predicate",
    "PredicateAnswer",
    "Question",
    "Refusal",
    "Score",
    "ScoreAnswer",
    "build_decision_input",
    "normalize_questions",
    "parse_decision",
    "question_from_dict",
    "summarize_decision_input",
]
