"""Agent tool-calling loop executor."""

from __future__ import annotations

import json
from typing import Any

from fastaiagent._internal.errors import MaxIterationsError, ToolExecutionError
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import (
    AssistantMessage,
    Message,
    ToolMessage,
)
from fastaiagent.tool.base import Tool


async def execute_tool_loop(
    llm: Any,
    messages: list[Message],
    tools: list[Tool],
    max_iterations: int = 10,
    tool_choice: str = "auto",
    tracer: Any = None,
) -> tuple[LLMResponse, list[dict]]:
    """Execute the agent's tool-calling loop.

    Sends messages to the LLM. If the LLM requests tool calls,
    executes them, appends results, and loops. Stops when the LLM
    returns a final response (no tool calls) or max_iterations is reached.

    Returns:
        Tuple of (final LLM response, list of all tool call records)
    """
    tool_defs = [t.to_openai_format() for t in tools] if tools else None
    tools_by_name = {t.name: t for t in tools}
    all_tool_calls: list[dict] = []

    for iteration in range(max_iterations):
        # Call LLM
        response = await llm.acomplete(messages, tools=tool_defs)

        # No tool calls — we're done
        if not response.tool_calls:
            return response, all_tool_calls

        # Build assistant message with tool calls
        messages.append(
            AssistantMessage(content=response.content, tool_calls=response.tool_calls)
        )

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
                    result = await tool.aexecute(tc.arguments)
                    if result.success:
                        result_text = (
                            json.dumps(result.output, default=str)
                            if not isinstance(result.output, str)
                            else result.output
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
        f"Agent exceeded maximum iterations ({max_iterations})"
    )
