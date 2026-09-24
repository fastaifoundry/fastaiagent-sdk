"""Say that a run paused, for callers that cannot hold the pause themselves.

Since 1.74.0 ``Agent.arun()`` returns a pause (``status="paused"``, ``output=""``)
when a managed approval policy or an ``interrupt()`` stops a tool call, and an
agent with no checkpointer raises the bare :class:`InterruptSignal` instead. A
caller that reads ``.output`` takes that ``""`` for the agent's answer. The MCP
server, ``simulate()``, ``evaluate()``, the pytest eval plugin and replay each
report the pause in their own "did not finish" form instead, with the text built
here so all of them say the same thing.

The text never includes the paused call's arguments (``tool_input``): they can be
sensitive, and some of these surfaces leave the process.
"""

from __future__ import annotations

from typing import Any


def describe_pause(obj: Any) -> str | None:
    """One line saying what a run is waiting on, or ``None`` if it finished.

    ``obj`` is a result with ``status`` / ``pending_interrupt`` / ``execution_id``
    (``AgentResult``, ``ChainResult``) or a raised ``InterruptSignal``.
    """
    from fastaiagent.chain.interrupt import InterruptSignal

    if isinstance(obj, InterruptSignal):
        return (
            f"paused: {_what(obj.reason, obj.context)}. The agent has no checkpointer, "
            f"so the pause could not be saved and the run cannot be resumed; give the "
            f"agent a checkpointer to hold pauses."
        )
    if getattr(obj, "status", None) != "paused":
        return None
    pending = getattr(obj, "pending_interrupt", None) or {}
    run_id = getattr(obj, "execution_id", None) or "?"
    return (
        f"paused: {_what(pending.get('reason'), pending.get('context'))}; "
        f"execution_id={run_id}. Nothing after the pause ran. Resolve it with "
        f"aresume({run_id!r}, resume_value=Resume(...)) or "
        f"`fastaiagent resume {run_id} --runner module:attr`."
    )


def what_paused(obj: Any) -> str | None:
    """Just the reason (and tool) a run paused on, or ``None`` if it did not."""
    from fastaiagent.chain.interrupt import InterruptSignal

    if isinstance(obj, InterruptSignal):
        return _what(obj.reason, obj.context)
    if getattr(obj, "status", None) != "paused":
        return None
    pending = getattr(obj, "pending_interrupt", None) or {}
    return _what(pending.get("reason"), pending.get("context"))


def _what(reason: Any, context: Any) -> str:
    tool = context.get("tool") if isinstance(context, dict) else None
    return f"{reason} (tool {tool!r})" if tool else str(reason)
