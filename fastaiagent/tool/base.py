"""Tool base class and ToolResult."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from fastaiagent._internal.async_utils import run_sync
from fastaiagent._internal.errors import ToolExecutionError

logger = logging.getLogger(__name__)

# Replay-safety classes drive the central Replay engine's inject-vs-execute
# decision per tool call. ``side_effecting`` is the safe default: an unmarked
# tool is never re-executed during replay (its recorded output is injected).
# Marks are explicit only — a "GET" REST tool is NOT auto-classified read_only;
# auto-inferring a re-executable class would violate the replay-safety invariant.
_ALLOWED_REPLAY_CLASSES = ("read_only", "idempotent", "side_effecting")
_DEFAULT_REPLAY_CLASS = "side_effecting"


class ToolResult(BaseModel):
    """Result of a tool execution."""

    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None


class Tool:
    """Base class for all tools.

    Subclasses: FunctionTool, RESTTool, MCPTool.
    """

    # Display origin for the Local UI. Subclasses override to one of
    # "function" / "mcp" / "rest". ``LocalKB.as_tool()`` overrides the
    # instance to "kb". Anything left at "custom" is a user-defined Tool
    # subclass. Surfaces on /agents/<name> as a colored chip so users can
    # tell at a glance whether a tool came from a decorator, an MCP server,
    # a REST spec, a knowledge base, or their own code.
    origin: str = "custom"

    def __init__(
        self,
        name: str,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        replay_class: str | None = None,
        *,
        timeout: float | None = None,
        max_retries: int = 0,
        retry_delay: float = 0.5,
        output_type: Any | None = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters or {"type": "object", "properties": {}}
        # Resolve the safe default *before* validating, so unset always passes.
        # Strict: an explicit out-of-set value is a developer error and raises
        # loudly here (the authoring boundary). The wire/replay layer stays
        # lenient and coerces unknown values to ``side_effecting`` at read time.
        resolved = _DEFAULT_REPLAY_CLASS if replay_class is None else replay_class
        if resolved not in _ALLOWED_REPLAY_CLASSES:
            raise ValueError(
                f"replay_class must be one of {_ALLOWED_REPLAY_CLASSES}, "
                f"got {replay_class!r}"
            )
        self.replay_class = resolved

        # Execution policy — applied by :meth:`ainvoke` (the agent-loop entry
        # point). All-unset is a no-op: ``ainvoke`` reduces to ``aexecute``.
        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be > 0 or None, got {timeout!r}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries!r}")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.output_type = output_type
        self._output_adapter: TypeAdapter[Any] | None = None

    def execute(self, arguments: dict[str, Any], context: Any | None = None) -> ToolResult:
        """Execute the tool synchronously (applies the tool's execution policy)."""
        return run_sync(self.ainvoke(arguments, context=context))

    async def aexecute(
        self,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> ToolResult:
        """Execute the tool asynchronously. Override in subclasses.

        This is the *raw* execution — it does not apply the timeout / retry /
        output-validation policy. The agent loop calls :meth:`ainvoke`, which
        wraps this; call ``aexecute`` directly only to deliberately bypass the
        policy.
        """
        raise NotImplementedError("Subclasses must implement aexecute()")

    async def ainvoke(
        self,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> ToolResult:
        """Execute with the tool's timeout / retry / output-validation policy.

        Wraps :meth:`aexecute`. With ``timeout``/``max_retries``/``output_type``
        all unset this is exactly ``await self.aexecute(...)``. Control-flow
        signals (``interrupt()`` etc.) always propagate and are never retried.

        A call that exhausts its retries re-raises the last error (so the agent
        loop's existing ``ToolExecutionError`` handling is preserved) or returns
        the last error :class:`ToolResult` if the tool signalled failure that
        way. On success, the output is validated/coerced against ``output_type``
        when set.
        """
        # Imported lazily to avoid a circular import at module load.
        from fastaiagent._internal.errors import StopAgent
        from fastaiagent.agent.executor import _AgentInterrupted
        from fastaiagent.chain.interrupt import AlreadyResumed, InterruptSignal

        control_flow = (InterruptSignal, _AgentInterrupted, AlreadyResumed, StopAgent)

        attempts = self.max_retries + 1
        last_exc: BaseException | None = None
        last_error_result: ToolResult | None = None

        for attempt in range(attempts):
            last_exc = None
            last_error_result = None
            try:
                coro = self.aexecute(arguments, context=context)
                if self.timeout is not None:
                    result = await asyncio.wait_for(coro, self.timeout)
                else:
                    result = await coro
            except control_flow:
                raise
            except (asyncio.TimeoutError, TimeoutError):
                # ``asyncio.wait_for`` raises ``asyncio.TimeoutError``, which is
                # a distinct class from the builtin ``TimeoutError`` before
                # Python 3.11 — catch both so timeouts are reported uniformly.
                last_exc = ToolExecutionError(
                    f"Tool '{self.name}' timed out after {self.timeout}s"
                )
            except Exception as e:  # noqa: BLE001 — retried/re-raised below
                last_exc = e
            else:
                if result.success:
                    return self._validate_output(result)
                last_error_result = result

            # Reached only on failure — back off and retry if attempts remain.
            if attempt < attempts - 1:
                await asyncio.sleep(self.retry_delay * (2**attempt))
                continue

        if last_error_result is not None:
            return last_error_result
        assert last_exc is not None
        raise last_exc

    def _validate_output(self, result: ToolResult) -> ToolResult:
        """Validate/coerce a successful result's output against ``output_type``.

        No-op when ``output_type`` is unset. On mismatch, returns an error
        ``ToolResult`` so the model sees the problem and can react — the tool's
        raw output never silently violates the declared schema.
        """
        if self.output_type is None:
            return result
        if self._output_adapter is None:
            self._output_adapter = TypeAdapter(self.output_type)
        try:
            coerced = self._output_adapter.validate_python(result.output)
        except ValidationError as e:
            logger.debug("Tool %r output failed validation", self.name, exc_info=True)
            return ToolResult(
                error=f"Tool '{self.name}' output failed schema validation: {e}",
                metadata=result.metadata,
            )
        return ToolResult(output=coerced, metadata=result.metadata)

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_mcp_schema(self) -> dict[str, Any]:
        """Convert to MCP tool-schema shape (``name`` / ``description`` / ``inputSchema``).

        Used by :class:`fastaiagent.tool.mcp_server.FastAIAgentMCPServer` when
        an agent's tools are exposed individually (``expose_tools=True``).
        """
        params = self.parameters or {"type": "object", "properties": {}}
        if "type" not in params:
            params = {"type": "object", **params}
        return {
            "name": self.name,
            "description": self.description or self.name,
            "inputSchema": params,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to canonical format."""
        return {
            "name": self.name,
            "description": self.description,
            "tool_type": self._tool_type(),
            "origin": self.origin,
            "parameters": self.parameters,
            "replay_class": self.replay_class,
            "config": self._config_dict(),
        }

    def _tool_type(self) -> str:
        return "base"

    def _config_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tool:
        """Deserialize from canonical format — dispatches to correct subclass."""
        from fastaiagent.tool.function import FunctionTool
        from fastaiagent.tool.mcp import MCPTool
        from fastaiagent.tool.rest import RESTTool

        tool_type = data.get("tool_type", "function")
        dispatch: dict[str, type[Tool]] = {
            "function": FunctionTool,
            "rest_api": RESTTool,
            "mcp": MCPTool,
        }
        target_cls = dispatch.get(tool_type)
        if target_cls is None:
            return cls(
                name=data["name"],
                description=data.get("description", ""),
                parameters=data.get("parameters"),
                replay_class=data.get("replay_class", _DEFAULT_REPLAY_CLASS),
            )
        result: Tool = target_cls._from_dict(data)  # type: ignore[attr-defined]
        return result
