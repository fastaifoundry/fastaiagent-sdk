"""Guardrail class and related types."""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from fastaiagent._internal.async_utils import run_sync


class GuardrailPosition(str, Enum):
    """Where in the pipeline this guardrail runs."""

    input = "input"
    output = "output"
    tool_call = "tool_call"
    tool_result = "tool_result"


class GuardrailType(str, Enum):
    """Implementation type of the guardrail."""

    code = "code"
    llm_judge = "llm_judge"
    regex = "regex"
    schema = "schema"
    classifier = "classifier"
    content_safety = "content_safety"
    groundedness = "groundedness"
    topic = "topic"
    pii = "pii"
    secrets = "secrets"


class GuardrailResult(BaseModel):
    """Result of a guardrail execution."""

    passed: bool
    score: float | None = None
    message: str | None = None
    execution_time_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    errored: bool = False
    """True when the check itself failed to run (e.g. the model call raised).

    ``passed`` then reflects the guardrail's ``on_error`` policy rather than a
    real verdict: ``on_error="allow"`` yields ``passed=True`` (fail open),
    ``on_error="block"`` yields ``passed=False`` (fail closed). Lets callers and
    the Local UI distinguish a degraded pass from a genuine one.
    """

    action: str = "block"
    """The consequence this guardrail was *configured* to carry. Never branch on
    this — branch on :attr:`action_taken`. See ``fastaiagent.guardrail.actions``."""

    action_taken: str = "none"
    """What the action actually did: ``none`` (clean pass) | ``blocked`` |
    ``warned`` | ``masked`` | ``overridden`` | ``reask``.

    Distinct from ``action`` because an action does not always get what it asked
    for: an errored check always blocks, and a ``mask`` that finds no span to
    redact degrades to a block rather than passing the payload through.
    """

    modified_data: str | dict[str, Any] | None = None
    """The rewritten payload when ``action_taken`` is ``masked`` or
    ``overridden``; ``None`` otherwise. The caller decides whether it can apply
    the rewrite faithfully — where it cannot, the outcome degrades to a block."""

    def rewrote_payload(self) -> bool:
        """True when this result carries a payload the caller should use instead."""
        return self.action_taken in ("masked", "overridden") and self.modified_data is not None


class Guardrail:
    """A validation guardrail for agent input/output/tool calls.

    Supports 10 implementation types: code, llm_judge, regex, schema, classifier,
    content_safety, groundedness, topic, pii, secrets.
    """

    def __init__(
        self,
        name: str,
        guardrail_type: GuardrailType = GuardrailType.code,
        position: GuardrailPosition = GuardrailPosition.output,
        config: dict[str, Any] | None = None,
        blocking: bool = True,
        description: str = "",
        fn: Callable[..., Any] | None = None,
        on_error: Literal["allow", "block"] = "block",
        origin: Literal["local", "plane"] = "local",
        action: str = "block",
        severity: str | None = None,
        floor: bool = False,
    ):
        self.name = name
        self.guardrail_type = guardrail_type
        self.position = position
        self.config = config or {}
        self.blocking = blocking
        self.description = description
        self.fn = fn  # for code guardrails with inline function
        # What to do when the check itself errors (e.g. a model-judged
        # detector's LLM call raises). "block" fails closed (default),
        # "allow" fails open. See GuardrailResult.errored.
        self.on_error: Literal["allow", "block"] = on_error
        # Who authored this guardrail. ``"plane"`` marks one reconstructed from
        # a control-plane policy rule (see guardrail.from_policy). It is a
        # *runtime* marker, deliberately absent from to_dict(): a plane-authored
        # guardrail must never be pushed back up as part of an agent's own
        # definition, or the plane would link it to that agent and thereby
        # narrow a domain-wide rule to just the agents that echoed it back.
        self.origin: Literal["local", "plane"] = origin
        # What a genuine failure costs: block | warn | mask | override | reask.
        # A third axis, independent of ``blocking`` (does it run inline and can
        # it halt?) and ``on_error`` (what does an un-runnable check mean?).
        # Coerced on the way in so an unrecognised value fails closed to
        # "block" rather than quietly letting the payload through.
        from fastaiagent.guardrail.actions import coerce_action, coerce_severity

        self.action: str = coerce_action(action)
        # Operator-assigned impact. Carried, shown and traced; nothing in the
        # SDK's enforcement depends on it.
        self.severity: str | None = coerce_severity(severity)
        # True when this is the domain-wide baseline only an admin may change.
        # Enforced by the plane; at the edge it is worth showing, because "this
        # is not your team's rule to argue with" is useful context locally.
        self.floor: bool = bool(floor)

    def execute(self, data: str | dict[str, Any]) -> GuardrailResult:
        """Execute the guardrail synchronously."""
        return run_sync(self.aexecute(data))

    async def aexecute(self, data: str | dict[str, Any]) -> GuardrailResult:
        """Execute the guardrail asynchronously."""
        import time

        from fastaiagent.guardrail.implementations import run_guardrail

        start = time.monotonic()
        result = await run_guardrail(self, data)
        result.execution_time_ms = int((time.monotonic() - start) * 1000)

        from fastaiagent._internal.config import get_config

        if get_config().ui_enabled:
            from fastaiagent.ui.events import log_guardrail_event

            try:
                log_guardrail_event(self, result, data=data)
            except Exception:
                # Logging is best-effort; never fail a guardrail check because the
                # event store hiccupped. ``log_guardrail_event`` is try/*finally*
                # with no ``except``, so a locked or read-only ``local.db``, a full
                # disk, or metadata that would not serialize used to propagate out
                # of here and abort the whole agent run — turning an observability
                # problem into an outage. The three framework integrations already
                # guarded this call; the SDK's own runtime did not.
                logging.getLogger(__name__).debug(
                    "Failed to log guardrail event for %r", self.name, exc_info=True
                )
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize to canonical format."""
        return {
            "name": self.name,
            "guardrail_type": self.guardrail_type.value,
            "position": self.position.value,
            "config": self.config,
            "blocking": self.blocking,
            "description": self.description,
            "on_error": self.on_error,
            "action": self.action,
            "severity": self.severity,
            "floor": self.floor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Guardrail:
        """Deserialize from canonical format."""
        return cls(
            name=data["name"],
            guardrail_type=GuardrailType(data.get("guardrail_type", "code")),
            position=GuardrailPosition(data.get("position", "output")),
            config=data.get("config", {}),
            blocking=data.get("blocking", True),
            description=data.get("description", ""),
            on_error=data.get("on_error", "block"),
            action=data.get("action", "block"),
            severity=data.get("severity"),
            floor=data.get("floor", False),
        )
