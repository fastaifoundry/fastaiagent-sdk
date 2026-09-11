"""Guardrail class and related types."""

from __future__ import annotations

import contextvars
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


class GuardrailFiring(BaseModel):
    """One guardrail execution, reported back on :attr:`AgentResult.guardrails`.

    Deliberately narrower than :class:`GuardrailResult`: no ``modified_data``
    and no ``metadata``. Those carry payload-derived content — a ``pii`` rule's
    ``matches`` holds the matched value itself — and this record exists to
    answer *did a rule fire, where, and what did it do*, not to re-expose the
    payload beside the answer.

    ``message`` is the exception, and it is diagnostic only: for ``regex`` it is
    the rule's own pattern and for ``llm_judge`` the judge's reply, so treat it
    as developer-facing. It never leaves the process — this object is returned
    to the caller in-memory and is not part of any span or wire payload.
    """

    name: str
    position: str
    action_taken: str
    """``none`` | ``blocked`` | ``warned`` | ``masked`` | ``overridden`` | ``reask``."""

    passed: bool
    errored: bool
    message: str | None = None

    def fired(self) -> bool:
        """True when this rule did something other than pass cleanly."""
        return self.action_taken != "none" or not self.passed or self.errored


#: Run-scoped collector for the above, appended by :meth:`Guardrail.aexecute`.
#: The agent installs a fresh list per run and hands it back on ``AgentResult``.
#:
#: ``None`` when nothing is collecting, which is the case for a bare
#: ``guardrail.execute(...)`` call — there is no run to attribute it to.
#:
#: Deliberately **not** behind ``ui_enabled``: the defect this closes is that an
#: unconnected run with the Local UI off had no way to tell that a ``warn`` or
#: ``mask`` rule had fired at all. ``AgentResult`` returned the same clean string
#: either way, so the one guardrail outcome designed *not* to stop the run was
#: also the one the caller could not observe.
_run_firings: contextvars.ContextVar[list[GuardrailFiring] | None] = contextvars.ContextVar(
    "fastaiagent_guardrail_firings", default=None
)


def start_firing_collection() -> contextvars.Token[list[GuardrailFiring] | None]:
    """Begin collecting guardrail firings for one run. Returns the reset token."""
    return _run_firings.set([])


def collected_firings() -> list[GuardrailFiring]:
    """The firings recorded since :func:`start_firing_collection`, oldest first."""
    return list(_run_firings.get() or [])


def stop_firing_collection(token: contextvars.Token[list[GuardrailFiring] | None]) -> None:
    """End the collection scope opened by :func:`start_firing_collection`."""
    _run_firings.reset(token)


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

        # Record the firing for whoever is collecting this run. Before the
        # ``ui_enabled`` gate below on purpose — this is the path that makes a
        # non-blocking outcome visible when there is no UI and no plane.
        firings = _run_firings.get()
        if firings is not None:
            try:
                firings.append(
                    GuardrailFiring(
                        name=self.name,
                        position=getattr(self.position, "value", str(self.position)),
                        action_taken=result.action_taken,
                        passed=result.passed,
                        errored=result.errored,
                        message=result.message,
                    )
                )
            except Exception:
                # Same rule as the event logger below: bookkeeping never fails
                # the check it is bookkeeping for.
                logging.getLogger(__name__).debug(
                    "Failed to record guardrail firing for %r", self.name, exc_info=True
                )

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
        """Deserialize from canonical format.

        A ``code`` guardrail carries its logic in ``fn=``, which cannot be
        serialized. For the **builtins** that is recoverable, because they are a
        closed set of zero-argument factories identified by name — so
        ``no_pii``, ``no_secrets``, ``toxicity_check``, ``json_valid``,
        ``no_prompt_injection`` and ``openai_moderation`` come back **armed**.
        That is what makes ``Replay.fork_at(...).rerun()`` a faithful
        reproduction rather than a rerun with the safety controls off.

        Everything else — a user's own ``fn=``, and the parameterised builtins
        ``cost_limit`` / ``allowed_domains`` / ``grounded`` whose policy lives in
        constructor arguments ``to_dict()`` never carried — comes back without a
        callable and **raises at execution time** (since 1.64.0) rather than
        reporting a clean pass over a payload nothing inspected.
        """
        guardrail_type = GuardrailType(data.get("guardrail_type", "code"))

        fn = None
        if guardrail_type is GuardrailType.code:
            from fastaiagent.guardrail.builtins import restore_builtin_fn

            fn = restore_builtin_fn(data["name"])

        return cls(
            name=data["name"],
            guardrail_type=guardrail_type,
            position=GuardrailPosition(data.get("position", "output")),
            config=data.get("config", {}),
            blocking=data.get("blocking", True),
            description=data.get("description", ""),
            on_error=data.get("on_error", "block"),
            action=data.get("action", "block"),
            severity=data.get("severity"),
            floor=data.get("floor", False),
            fn=fn,
        )
