"""Tool system — FunctionTool, RESTTool, MCPTool."""

from fastaiagent.tool.base import Tool, ToolResult
from fastaiagent.tool.function import FunctionTool, tool
from fastaiagent.tool.mcp import MCPTool
from fastaiagent.tool.registry import ToolRegistry
from fastaiagent.tool.rest import RESTTool

__all__ = [
    "Tool",
    "ToolResult",
    "FunctionTool",
    "RESTTool",
    "MCPTool",
    "ToolRegistry",
    "tool",
]
