"""Agent tool-calling loop executor."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from fastaiagent._internal.errors import MaxIterationsError, StopAgent, ToolExecutionError
from fastaiagent.agent.middleware import MiddlewareContext, _MiddlewarePipeline
from fastaiagent.guardrail.executor import execute_guardrails
from fastaiagent.guardrail.guardrail import GuardrailPosition
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
)
from fastaiagent.llm.stream import (
    StreamEvent,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
)
from fastaiagent.tool.base import Tool, ToolResult

if TYPE_CHECKING:
    from fastaiagent.guardrail.guardrail import Guardrail


class _StubTool(Tool):
    """Placeholder Tool used when the LLM requests a tool that is not registered.

    Passed into ``wrap_tool`` middleware so that middleware always sees a Tool
    object. The ``aexecute`` path is never reached — the terminal closure
    handles the unknown-tool branch directly.
    """

    def __init__(self, name: str):
        super().__init__(name=name, description="(unknown tool)")

    async def aexecute(
        self, arguments: dict[str, Any], context: Any | None = None
    ) -> ToolResult:
        return ToolResult(error=f"Unknown tool '{self.name}'")


async def _invoke_tool_with_span(
    tool: Tool | None,
    tool_name: str,
    arguments: dict[str, Any],
    context: Any | None,
    guardrails: list[Guardrail] | None,
    tool_call_record: dict[str, Any] | None = None,
) -> str:
    """Execute a single tool call inside an OTel span and return the textual result.

    Always creates a span (even for unknown tools) so dashboards see the attempt.
    Captures tool.name and tool.status unconditionally; tool.args / tool.result
    are gated by ``trace_payloads_enabled()``.
    """
    from fastaiagent.trace.otel import get_tracer
    from fastaiagent.trace.span import trace_payloads_enabled

    tracer = get_tracer("fastaiagent.agent.executor")
    with tracer.start_as_current_span(f"tool.{tool_name}") as span:
        span.set_attribute("tool.name", tool_name)
        # Origin lets the UI group "function / kb / mcp / rest / custom"
        # without having to cross-reference agent.tools. "unknown" when the
        # LLM hallucinates a tool that isn't registered.
        span.set_attribute(
            "tool.origin", getattr(tool, "origin", "unknown") if tool else "unknown"
        )
        if trace_payloads_enabled():
            try:
                span.set_attribute("tool.args", json.dumps(arguments, default=str))
            except Exception:
                pass

        if tool is None:
            result_text = f"Error: Unknown tool '{tool_name}'"
            span.set_attribute("tool.status", "unknown")
            if tool_call_record is not None:
                tool_call_record["error"] = result_text
            return result_text

        try:
            # Tool-call guardrail: validate arguments before execution
            if guardrails:
                tc_data = json.dumps(
                    {"tool": tool_name, "arguments": arguments}, default=str
                )
                await execute_guardrails(guardrails, tc_data, GuardrailPosition.tool_call)

            result = await tool.aexecute(arguments, context=context)
            if result.success:
                result_text = (
                    json.dumps(result.output, default=str)
                    if not isinstance(result.output, str)
                    else result.output
                )
                # Tool-result guardrail: validate output after execution
                if guardrails:
                    await execute_guardrails(
                        guardrails, result_text, GuardrailPosition.tool_result
                    )
                span.set_attribute("tool.status", "ok")
            else:
                result_text = f"Error: {result.error}"
                span.set_attribute("tool.status", "error")
                span.set_attribute("tool.error", str(result.error))

            if trace_payloads_enabled():
                span.set_attribute("tool.result", result_text)
            if tool_call_record is not None:
                tool_call_record["output"] = result_text
            return result_text
        except ToolExecutionError as e:
            result_text = f"Error: {e}"
            span.set_attribute("tool.status", "error")
            span.set_attribute("tool.error", str(e))
            if tool_call_record is not None:
                tool_call_record["error"] = str(e)
            return result_text


async def execute_tool_loop(
    llm: Any,
    messages: list[Message],
    tools: list[Tool],
    max_iterations: int = 10,
    tool_choice: str = "auto",
    tracer: Any = None,
    context: Any | None = None,
    guardrails: list[Guardrail] | None = None,
    mw_pipeline: _MiddlewarePipeline | None = None,
    mw_ctx: MiddlewareContext | None = None,
    **kwargs: Any,
) -> tuple[LLMResponse, list[dict[str, Any]]]:
    """Execute the agent's tool-calling loop.

    Sends messages to the LLM. If the LLM requests tool calls,
    executes them, appends results, and loops. Stops when the LLM
    returns a final response (no tool calls) or max_iterations is reached.

    When ``mw_pipeline`` and ``mw_ctx`` are provided, middleware hooks fire:
    ``before_model`` before each LLM call, ``after_model`` after each LLM
    response, and ``wrap_tool`` around each tool invocation.

    Returns:
        Tuple of (final LLM response, list of all tool call records)
    """
    tool_defs = [t.to_openai_format() for t in tools] if tools else None
    tools_by_name = {t.name: t for t in tools}
    all_tool_calls: list[dict[str, Any]] = []

    for iteration in range(max_iterations):
        # Middleware: before_model (may raise StopAgent)
        if mw_pipeline and mw_ctx is not None:
            mw_ctx.turn = iteration
            try:
                messages = await mw_pipeline.apply_before_model(mw_ctx, messages)
            except StopAgent as stop:
                return LLMResponse(content=str(stop), finish_reason="stop"), all_tool_calls

        # Call LLM
        response = await llm.acomplete(messages, tools=tool_defs, **kwargs)

        # Middleware: after_model (may raise StopAgent)
        if mw_pipeline and mw_ctx is not None:
            try:
                response = await mw_pipeline.apply_after_model(mw_ctx, response)
            except StopAgent as stop:
                return LLMResponse(content=str(stop), finish_reason="stop"), all_tool_calls

        # No tool calls — we're done
        if not response.tool_calls:
            return response, all_tool_calls

        # Build assistant message with tool calls
        messages.append(AssistantMessage(content=response.content, tool_calls=response.tool_calls))

        # Execute each tool call
        for idx, tc in enumerate(response.tool_calls):
            tool_call_record = {
                "iteration": iteration,
                "tool_name": tc.name,
                "arguments": tc.arguments,
                "tool_call_id": tc.id,
            }

            tool = tools_by_name.get(tc.name)

            if mw_pipeline and mw_ctx is not None:
                mw_ctx.tool_call_index = idx

                async def _terminal(
                    t: Tool,
                    a: dict[str, Any],
                    _tc: Any = tc,
                    _record: dict[str, Any] = tool_call_record,
                ) -> ToolResult:
                    text = await _invoke_tool_with_span(
                        tool=tools_by_name.get(_tc.name),
                        tool_name=_tc.name,
                        arguments=a,
                        context=context,
                        guardrails=guardrails,
                        tool_call_record=_record,
                    )
                    return ToolResult(output=text)

                wrap_target = tool if tool is not None else _StubTool(tc.name)
                try:
                    tr = await mw_pipeline.invoke_tool(
                        mw_ctx, wrap_target, dict(tc.arguments), _terminal
                    )
                except StopAgent as stop:
                    # Include the in-flight tool-call record; the terminal
                    # closure already populated ``output``/``error`` via
                    # ``_invoke_tool_with_span`` before the stopper fired.
                    if tool_call_record not in all_tool_calls:
                        all_tool_calls.append(tool_call_record)
                    return (
                        LLMResponse(content=str(stop), finish_reason="stop"),
                        all_tool_calls,
                    )
                if isinstance(tr.output, str):
                    result_text = tr.output
                elif tr.error is not None:
                    result_text = f"Error: {tr.error}"
                else:
                    result_text = json.dumps(tr.output, default=str)
            else:
                result_text = await _invoke_tool_with_span(
                    tool=tool,
                    tool_name=tc.name,
                    arguments=tc.arguments,
                    context=context,
                    guardrails=guardrails,
                    tool_call_record=tool_call_record,
                )

            messages.append(ToolMessage(content=result_text, tool_call_id=tc.id))
            all_tool_calls.append(tool_call_record)

    raise MaxIterationsError(
        f"Agent exceeded maximum iterations ({max_iterations}). "
        f"The LLM continued requesting tool calls beyond the limit.\n"
        f"Options:\n"
        f"  1. Increase the limit: AgentConfig(max_iterations={max_iterations * 2})\n"
        f"  2. Review the system prompt to ensure the agent can reach a final answer\n"
        f"  3. Simplify the available tools to reduce unnecessary tool-calling loops"
    )


async def stream_tool_loop(
    llm: Any,
    messages: list[Message],
    tools: list[Tool],
    max_iterations: int = 10,
    tool_choice: str = "auto",
    context: Any | None = None,
    guardrails: list[Guardrail] | None = None,
    **kwargs: Any,
) -> AsyncGenerator[StreamEvent, None]:
    """Streaming version of execute_tool_loop.

    Yields StreamEvent objects as tokens arrive from the LLM.
    Handles tool execution between streaming iterations.

    The final TextDelta events contain the agent's response text.
    ToolCallStart/ToolCallEnd events are emitted for both LLM-requested
    tool calls and their execution results.
    """
    tool_defs = [t.to_openai_format() for t in tools] if tools else None
    tools_by_name = {t.name: t for t in tools}

    for iteration in range(max_iterations):
        # Stream from LLM
        accumulated_text = ""
        pending_tool_calls: list[ToolCall] = []
        total_usage = Usage()

        async for event in llm.astream(messages, tools=tool_defs, **kwargs):
            if isinstance(event, TextDelta):
                accumulated_text += event.text
                yield event
            elif isinstance(event, ToolCallStart):
                yield event
            elif isinstance(event, ToolCallEnd):
                pending_tool_calls.append(
                    ToolCall(id=event.call_id, name=event.tool_name, arguments=event.arguments)
                )
                yield event
            elif isinstance(event, Usage):
                total_usage = event
                yield event
            # Don't yield StreamDone here — we may have more iterations

        # No tool calls — final response, we're done
        if not pending_tool_calls:
            return

        # Append assistant message with accumulated content + tool calls
        messages.append(
            AssistantMessage(
                content=accumulated_text or None,
                tool_calls=pending_tool_calls,
            )
        )

        # Execute each tool call
        for tc in pending_tool_calls:
            tool = tools_by_name.get(tc.name)
            result_text = await _invoke_tool_with_span(
                tool=tool,
                tool_name=tc.name,
                arguments=tc.arguments,
                context=context,
                guardrails=guardrails,
            )

            messages.append(ToolMessage(content=result_text, tool_call_id=tc.id))

    raise MaxIterationsError(
        f"Agent exceeded maximum iterations ({max_iterations}). "
        f"The LLM continued requesting tool calls beyond the limit.\n"
        f"Options:\n"
        f"  1. Increase the limit: AgentConfig(max_iterations={max_iterations * 2})\n"
        f"  2. Review the system prompt to ensure the agent can reach a final answer\n"
        f"  3. Simplify the available tools to reduce unnecessary tool-calling loops"
    )
