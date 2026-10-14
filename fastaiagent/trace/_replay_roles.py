"""Recorded replay serves each model call from the recordings of its own role (1.89.0).

``with_determinism("recorded")`` installs the original trace's captured model
responses as a queue, and every ``LLMClient.acomplete`` in the rerun pops the
front. A guardrail's own model call (an ``llm_judge``, ``content_safety``,
``groundedness`` or ``topic`` check) is an ``acomplete`` too, so it popped from
the same queue. A plane-distributed judge present in the rerun but not in the
original took the agent's first turn, and every later turn got the one after
its own. A judge present in the original but not in the rerun handed its
verdict to the agent instead.

Now each call has a **role**:

* ``guardrail``: the call runs inside a ``guardrail.*`` span (since 1.89.0 a
  guardrail's span wraps its check, so a judge's model call is its child);
* ``agent``: every other call, served exactly as before.

The original trace's recordings are split the same way, by span ancestry. A
trace recorded before 1.89.0 has the judge's ``llm.*`` span beside the
guardrail span rather than under it; there a call is a guardrail's when the
trace has guardrail spans and the request is the shape every SDK guardrail
sends (one system message, then one user message that opens with the
``<<DATA>>`` / ``<<CONTEXT>>`` block).

At rerun time the role is read from the current span: a span processor marks
the ``guardrail.*`` spans the rerun opens and their descendants. A judge call
with no recording of its own is a miss handled by ``on_miss``. It never takes an
agent turn. With tracing switched off there are no spans to read the role
from, so the rerun falls back to the single shared queue and says so.

``acomplete`` / ``adecide`` are unchanged: :class:`RoleQueue` stands where the
list stood and answers ``len`` / ``[0]`` / ``del [0]`` / ``pop(0)`` from the
calling role's own list.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from typing import Any

from opentelemetry.sdk.trace import SpanProcessor

AGENT = "agent"
GUARDRAIL = "guardrail"

_GUARDRAIL_SPAN_PREFIX = "guardrail."
_GUARDRAIL_PAYLOAD_OPENERS = ("<<DATA>>", "<<CONTEXT>>")


class _Tracker:
    """The span ids a rerun has opened inside guardrail checks."""

    def __init__(self) -> None:
        self.guardrail_span_ids: set[int] = set()


_tracker: ContextVar[_Tracker | None] = ContextVar("fastaiagent_replay_roles", default=None)


class _RoleSpanProcessor(SpanProcessor):
    """Marks ``guardrail.*`` spans, and every span under one, while a rerun is active.

    ``on_start`` runs synchronously in the context of the code opening the span,
    so it sees the rerun's tracker and nothing outside a rerun pays more than
    one ContextVar read.
    """

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        tracker = _tracker.get()
        if tracker is None:
            return
        try:
            parent = getattr(span, "parent", None)
            if (getattr(span, "name", "") or "").startswith(_GUARDRAIL_SPAN_PREFIX) or (
                parent is not None and parent.span_id in tracker.guardrail_span_ids
            ):
                tracker.guardrail_span_ids.add(span.get_span_context().span_id)
        except Exception:  # pragma: no cover - bookkeeping never breaks a span
            return


_MARK = "_fastaiagent_replay_roles"


def role_tracking_available() -> bool:
    """Attach the role processor to the SDK's tracer provider, once per provider.

    ``False`` when tracing is off (the no-op provider opens no spans to read a
    role from).
    """
    from fastaiagent.trace.otel import get_tracer_provider

    provider = get_tracer_provider()
    if getattr(provider, _MARK, False):
        return True
    attach = getattr(provider, "add_span_processor", None)
    if attach is None:
        return False
    attach(_RoleSpanProcessor())
    try:
        setattr(provider, _MARK, True)
    except Exception:  # pragma: no cover - a provider that refuses attributes
        return False
    return True


def start_tracking() -> Any:
    """Begin marking guardrail spans for this rerun; returns a reset token."""
    return _tracker.set(_Tracker())


def stop_tracking(token: Any) -> None:
    _tracker.reset(token)


def current_role() -> str:
    """The role of the model call being made now."""
    tracker = _tracker.get()
    if tracker is None:
        return AGENT
    from opentelemetry import trace

    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid and ctx.span_id in tracker.guardrail_span_ids:
        return GUARDRAIL
    return AGENT


class RoleQueue:
    """The recorded-response queue, answered from the calling role's own list."""

    def __init__(self, by_role: dict[str, list[Any]]) -> None:
        self._by_role = by_role

    def _queue(self) -> list[Any]:
        return self._by_role.setdefault(current_role(), [])

    def __len__(self) -> int:
        return len(self._queue())

    def __getitem__(self, index: int) -> Any:
        return self._queue()[index]

    def __delitem__(self, index: int) -> None:
        del self._queue()[index]

    def pop(self, index: int = -1) -> Any:
        return self._queue().pop(index)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._queue())


# -- the recorded side ---------------------------------------------------------


def _looks_like_guardrail_request(span: Any) -> bool:
    """The request every SDK guardrail model call sends: system, then a user
    message opening with the payload block."""
    raw = (span.attributes or {}).get("gen_ai.request.messages")
    try:
        msgs = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return False
    if not isinstance(msgs, list) or len(msgs) != 2:
        return False
    first, second = msgs
    if not (isinstance(first, dict) and isinstance(second, dict)):
        return False
    if first.get("role") != "system" or second.get("role") != "user":
        return False
    content = second.get("content")
    return isinstance(content, str) and content.startswith(_GUARDRAIL_PAYLOAD_OPENERS)


def recorded_roles(spans: list[Any]) -> dict[str, str]:
    """``span_id -> role`` for every span of a recorded trace."""
    by_id = {s.span_id: s for s in spans}
    has_guardrails = any((s.name or "").startswith(_GUARDRAIL_SPAN_PREFIX) for s in spans)
    roles: dict[str, str] = {}
    for span in spans:
        role = AGENT
        seen: set[str] = set()
        parent_id = span.parent_span_id
        while parent_id and parent_id not in seen:
            seen.add(parent_id)
            parent = by_id.get(parent_id)
            if parent is None:
                break
            if (parent.name or "").startswith(_GUARDRAIL_SPAN_PREFIX):
                role = GUARDRAIL
                break
            parent_id = parent.parent_span_id
        if role == AGENT and has_guardrails and _looks_like_guardrail_request(span):
            role = GUARDRAIL
        roles[span.span_id] = role
    return roles


def split_by_role(spans: list[Any], recover: Callable[[Any], Any]) -> dict[str, list[Any]]:
    """Everything ``recover`` rebuilds from ``spans``, by role, in start order."""
    roles = recorded_roles(spans)
    out: dict[str, list[Any]] = {AGENT: [], GUARDRAIL: []}
    for span in sorted(spans, key=lambda s: s.start_time or ""):
        recovered = recover(span)
        if recovered is not None:
            out[roles.get(span.span_id, AGENT)].append(recovered)
    return out
