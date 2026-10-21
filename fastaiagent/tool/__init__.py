"""Tool system — FunctionTool, RESTTool, MCPTool, and provider-run tools."""

from fastaiagent.tool.base import Tool, ToolResult
from fastaiagent.tool.decision import decision_tool
from fastaiagent.tool.function import FunctionTool, tool
from fastaiagent.tool.mcp import MCPTool
from fastaiagent.tool.provider import ProviderTool, code_execution, file_search, web_search
from fastaiagent.tool.registry import ToolRegistry
from fastaiagent.tool.rest import RESTTool

__all__ = [
    "Tool",
    "ToolResult",
    "FunctionTool",
    "decision_tool",
    "RESTTool",
    "MCPTool",
    "ProviderTool",
    "ToolRegistry",
    "code_execution",
    "file_search",
    "tool",
    "web_search",
]
