"""Tests for fastaiagent.tool module."""

from __future__ import annotations

import pytest

from fastaiagent._internal.errors import ToolExecutionError
from fastaiagent.tool import FunctionTool, MCPTool, RESTTool, Tool, ToolResult, tool
from fastaiagent.tool.schema import DriftReport, detect_drift, validate_schema


# --- ToolResult tests ---


class TestToolResult:
    def test_success(self):
        r = ToolResult(output="hello")
        assert r.success is True

    def test_error(self):
        r = ToolResult(error="something broke")
        assert r.success is False


# --- FunctionTool tests ---


class TestFunctionTool:
    def test_from_callable(self):
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        t = FunctionTool(name="greet", fn=greet)
        assert t.name == "greet"
        assert "name" in t.parameters.get("properties", {})
        assert "name" in t.parameters.get("required", [])

    def test_execute(self):
        def add(a: int, b: int) -> int:
            return a + b

        t = FunctionTool(name="add", fn=add)
        result = t.execute({"a": 2, "b": 3})
        assert result.output == 5
        assert result.success

    @pytest.mark.asyncio
    async def test_aexecute(self):
        def multiply(x: int, y: int) -> int:
            return x * y

        t = FunctionTool(name="multiply", fn=multiply)
        result = await t.aexecute({"x": 4, "y": 5})
        assert result.output == 20

    @pytest.mark.asyncio
    async def test_async_function(self):
        async def fetch(url: str) -> str:
            return f"fetched: {url}"

        t = FunctionTool(name="fetch", fn=fetch)
        result = await t.aexecute({"url": "https://example.com"})
        assert result.output == "fetched: https://example.com"

    def test_execute_error_raises(self):
        def fail(x: str) -> str:
            raise ValueError("boom")

        t = FunctionTool(name="fail", fn=fail)
        with pytest.raises(ToolExecutionError, match="boom"):
            t.execute({"x": "test"})

    def test_auto_schema_optional_param(self):
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        t = FunctionTool(name="greet", fn=greet)
        assert "name" in t.parameters.get("required", [])
        assert "greeting" not in t.parameters.get("required", [])
        assert "greeting" in t.parameters.get("properties", {})

    def test_docstring_as_description(self):
        def search(query: str) -> str:
            """Search the knowledge base."""
            return query

        t = FunctionTool(name="search", fn=search)
        assert t.description == "Search the knowledge base."

    def test_to_openai_format(self):
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        t = FunctionTool(name="greet", fn=greet, description="Greet someone")
        fmt = t.to_openai_format()
        assert fmt["type"] == "function"
        assert fmt["function"]["name"] == "greet"
        assert fmt["function"]["description"] == "Greet someone"


# --- tool decorator tests ---


class TestToolDecorator:
    def test_basic_decorator(self):
        @tool(name="say_hello")
        def say_hello(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}!"

        assert isinstance(say_hello, FunctionTool)
        assert say_hello.name == "say_hello"
        result = say_hello.execute({"name": "World"})
        assert result.output == "Hello, World!"


# --- Tool serialization tests ---


class TestToolSerialization:
    def test_function_tool_to_dict(self):
        t = FunctionTool(
            name="test",
            description="A test tool",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
        )
        d = t.to_dict()
        assert d["name"] == "test"
        assert d["tool_type"] == "function"
        assert d["parameters"]["properties"]["x"]["type"] == "integer"

    def test_rest_tool_to_dict(self):
        t = RESTTool(
            name="weather",
            url="https://api.weather.com/forecast",
            method="GET",
            description="Get weather",
        )
        d = t.to_dict()
        assert d["tool_type"] == "rest_api"
        assert d["config"]["url"] == "https://api.weather.com/forecast"
        assert d["config"]["method"] == "GET"

    def test_mcp_tool_to_dict(self):
        t = MCPTool(
            name="search",
            server_url="http://localhost:3000",
            tool_name="search_files",
        )
        d = t.to_dict()
        assert d["tool_type"] == "mcp"
        assert d["config"]["server_url"] == "http://localhost:3000"

    def test_from_dict_dispatches_function(self):
        data = {
            "name": "test",
            "tool_type": "function",
            "description": "Test",
            "parameters": {"type": "object", "properties": {}},
        }
        t = Tool.from_dict(data)
        assert isinstance(t, FunctionTool)

    def test_from_dict_dispatches_rest(self):
        data = {
            "name": "api",
            "tool_type": "rest_api",
            "config": {"url": "https://example.com", "method": "POST"},
        }
        t = Tool.from_dict(data)
        assert isinstance(t, RESTTool)
        assert t.url == "https://example.com"

    def test_from_dict_dispatches_mcp(self):
        data = {
            "name": "mcp_tool",
            "tool_type": "mcp",
            "config": {"server_url": "http://localhost:3000", "tool_name": "search"},
        }
        t = Tool.from_dict(data)
        assert isinstance(t, MCPTool)
        assert t.server_url == "http://localhost:3000"


# --- Schema validation tests ---


class TestSchemaValidation:
    def test_valid_response(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
        violations = validate_schema(schema, {"name": "Alice", "age": 30})
        assert len(violations) == 0

    def test_wrong_type(self):
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        violations = validate_schema(schema, {"age": "not a number"})
        assert len(violations) == 1
        assert violations[0].field == "age"

    def test_missing_required(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        violations = validate_schema(schema, {})
        assert len(violations) == 1
        assert "missing" in violations[0].message.lower()

    def test_root_type_mismatch(self):
        schema = {"type": "object"}
        violations = validate_schema(schema, "not an object")
        assert len(violations) == 1

    def test_detect_drift_no_drift(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        report = detect_drift("test_tool", schema, [{"x": 1}, {"x": 2}, {"x": 3}])
        assert not report.drift_detected
        assert report.responses_checked == 3

    def test_detect_drift_with_drift(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        report = detect_drift("test_tool", schema, [{"x": 1}, {"x": "oops"}])
        assert report.drift_detected
        assert len(report.violations) == 1
