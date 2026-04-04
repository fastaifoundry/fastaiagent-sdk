"""Agent tool-calling loop executor."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from fastaiagent._internal.errors import MaxIterationsError, ToolExecutionError
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
from fastaiagent.tool.base import Tool

if TYPE_CHECKING:
    from fastaiagent.guardrail.guardrail import Guardrail


async def execute_tool_loop(
    llm: Any,
    messages: list[Message],
    tools: list[Tool],
    max_iterations: int = 10,
    tool_choice: str = "auto",
    tracer: Any = None,
    context: Any | None = None,
    guardrails: list[Guardrail] | None = None,
    **kwargs: Any,
) -> tuple[LLMResponse, list[dict[str, Any]]]:
    """Execute the agent's tool-calling loop.

    Sends messages to the LLM. If the LLM requests tool calls,
    executes them, appends results, and loops. Stops when the LLM
    returns a final response (no tool calls) or max_iterations is reached.

    Returns:
        Tuple of (final LLM response, list of all tool call records)
    """
    tool_defs = [t.to_openai_format() for t in tools] if tools else None
    tools_by_name = {t.name: t for t in tools}
    all_tool_calls: list[dict[str, Any]] = []

    for iteration in range(max_iterations):
        # Call LLM
        response = await llm.acomplete(messages, tools=tool_defs, **kwargs)

        # No tool calls — we're done
        if not response.tool_calls:
            return response, all_tool_calls

        # Build assistant message with tool calls
        messages.append(AssistantMessage(content=response.content, tool_calls=response.tool_calls))

        # Execute each tool call
        for tc in response.tool_calls:
            tool_call_record = {
                "iteration": iteration,
                "tool_name": tc.name,
                "arguments": tc.arguments,
                "tool_call_id": tc.id,
            }

            tool = tools_by_name.get(tc.name)
            if tool is None:
                result_text = f"Error: Unknown tool '{tc.name}'"
                tool_call_record["error"] = result_text
            else:
                try:
                    # Tool-call guardrail: validate arguments before execution
                    if guardrails:
                        tc_data = json.dumps(
                            {"tool": tc.name, "arguments": tc.arguments}, default=str
                        )
                        await execute_guardrails(
                            guardrails, tc_data, GuardrailPosition.tool_call
                        )

                    result = await tool.aexecute(tc.arguments, context=context)
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
                    else:
                        result_text = f"Error: {result.error}"
                    tool_call_record["output"] = result_text
                except ToolExecutionError as e:
                    result_text = f"Error: {e}"
                    tool_call_record["error"] = str(e)

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
            if tool is None:
                result_text = f"Error: Unknown tool '{tc.name}'"
            else:
                try:
                    # Tool-call guardrail: validate arguments before execution
                    if guardrails:
                        tc_data = json.dumps(
                            {"tool": tc.name, "arguments": tc.arguments}, default=str
                        )
                        await execute_guardrails(
                            guardrails, tc_data, GuardrailPosition.tool_call
                        )

                    result = await tool.aexecute(tc.arguments, context=context)
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
                    else:
                        result_text = f"Error: {result.error}"
                except ToolExecutionError as e:
                    result_text = f"Error: {e}"

            messages.append(ToolMessage(content=result_text, tool_call_id=tc.id))

    raise MaxIterationsError(
        f"Agent exceeded maximum iterations ({max_iterations}). "
        f"The LLM continued requesting tool calls beyond the limit.\n"
        f"Options:\n"
        f"  1. Increase the limit: AgentConfig(max_iterations={max_iterations * 2})\n"
        f"  2. Review the system prompt to ensure the agent can reach a final answer\n"
        f"  3. Simplify the available tools to reduce unnecessary tool-calling loops"
    )
