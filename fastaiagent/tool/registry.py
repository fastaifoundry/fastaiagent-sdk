"""Process-wide tool registry for replay reconstruction.

Tool function callables cannot be serialized into a trace, so when a replay
is rehydrated via ``Replay.load(trace_id).fork_at(step).rerun()`` the tool
names recovered from span attributes need to be rebound to live Python
functions. The registry holds those bindings.

Tools are auto-registered on creation by ``FunctionTool.__init__`` and by the
``@tool`` decorator, so existing user code "just works" when forked and rerun.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastaiagent.tool.base import Tool

_log = logging.getLogger(__name__)


class ToolRegistry:
    """Process-wide, name-keyed tool registry.

    Last-write-wins semantics: registering a tool with the same name as an
    existing entry replaces it. Callers that care about isolation should use
    distinct tool names.
    """

    _tools: dict[str, Tool] = {}

    @classmethod
    def register(cls, tool: Tool) -> Tool:
        """Register a tool. Returns the tool for decorator-friendly chaining.

        Inside a :func:`fastaiagent.runtime.job_scope` the tool is registered
        **job-locally** so it can't clobber a sibling job's same-named tool in
        the shared global registry.
        """
        from fastaiagent._internal.scope import scoped_tools

        scoped = scoped_tools.get()
        if scoped is not None:
            scoped[tool.name] = tool
        else:
            cls._tools[tool.name] = tool
        return tool

    @classmethod
    def get(cls, name: str) -> Tool | None:
        """Look up a tool by name. Returns None if not registered.

        Inside a ``job_scope`` the per-job tools are consulted first, then the
        global registry (overlay — a job's tool wins on a name collision).
        """
        from fastaiagent._internal.scope import scoped_tools

        scoped = scoped_tools.get()
        if scoped is not None and name in scoped:
            return scoped[name]
        return cls._tools.get(name)

    @classmethod
    def all(cls) -> dict[str, Tool]:
        """Return a copy of the registry contents (global registry overlaid by
        any active job scope)."""
        from fastaiagent._internal.scope import scoped_tools

        scoped = scoped_tools.get()
        if scoped is None:
            return dict(cls._tools)
        return {**cls._tools, **scoped}

    @classmethod
    def clear(cls) -> None:
        """Reset the registry. Intended for tests."""
        cls._tools.clear()

    @classmethod
    def unregister(cls, name: str) -> Tool | None:
        """Remove a tool by name. Returns the removed tool or None."""
        return cls._tools.pop(name, None)
