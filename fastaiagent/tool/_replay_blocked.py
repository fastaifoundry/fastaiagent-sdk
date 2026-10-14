"""The stand-in a replay rerun uses for a tool it may not run (1.89.0).

A rerun runs only the tools it was given (``with_tool_override``,
``with_tool_stub``) or allowed (``with_tool_policy``). Every other tool in the
original agent becomes one of these: it keeps the original's name, description
and parameter schema, so the model sees the same tool list, but calling it runs
nothing. The model gets an error result saying why and how to allow it, and the
call is counted.

It is never registered globally, and it serializes as the original tool (same
``tool_type`` and ``config``), so a replay of a replay can still rebuild it.
"""

from __future__ import annotations

from typing import Any

from fastaiagent.tool.base import Tool, ToolResult

_REPLAY_CLASSES = ("read_only", "idempotent", "side_effecting")


def _coerce_replay_class(value: Any) -> str:
    """The wire's replay_class, unknown values read as the safe default."""
    return value if value in _REPLAY_CLASSES else "side_effecting"


class ReplayBlockedTool(Tool):
    """A tool a replay rerun refuses to run. See the module docstring."""

    origin = "replay_blocked"

    def __init__(self, spec: dict[str, Any], *, reason: str) -> None:
        super().__init__(
            name=str(spec["name"]),
            description=spec.get("description", "") or "",
            parameters=spec.get("parameters"),
            replay_class=_coerce_replay_class(spec.get("replay_class")),
        )
        self._spec_tool_type = str(spec.get("tool_type", "function"))
        self._spec_config = dict(spec.get("config") or {})
        self.reason = reason
        self.calls = 0

    def _tool_type(self) -> str:
        return self._spec_tool_type

    def _config_dict(self) -> dict[str, Any]:
        return dict(self._spec_config)

    async def aexecute(self, arguments: dict[str, Any], context: Any | None = None) -> ToolResult:
        self.calls += 1
        name = self.name
        return ToolResult(
            error=(
                f"Replay blocked tool '{name}': {self.reason}. "
                f"Use with_tool_override('{name}', ...), "
                f"with_tool_stub('{name}', ...) or with_tool_policy('live', allow=['{name}'])."
            )
        )


__all__ = ["ReplayBlockedTool"]
