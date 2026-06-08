"""Tests for fastaiagent.tool module."""

from __future__ import annotations

import pytest

from fastaiagent._internal.errors import ToolExecutionError
from fastaiagent.tool import FunctionTool, MCPTool, RESTTool, Tool, ToolResult, tool
from fastaiagent.tool.schema import detect_drift, validate_schema

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

    def test_parses_google_docstring_params(self):
        def search(query: str, top_k: int = 5) -> str:
            """Search the knowledge base.

            Args:
                query: The search query to find relevant documents.
                top_k: Number of results to return.
            """
            return "results"

        t = FunctionTool(name="search", fn=search)
        props = t.parameters["properties"]
        assert props["query"]["description"] == "The search query to find relevant documents."
        assert props["top_k"]["description"] == "Number of results to return."

    def test_falls_back_to_param_name(self):
        def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}!"

        t = FunctionTool(name="greet", fn=greet)
        assert t.parameters["properties"]["name"]["description"] == "name"

    def test_handles_no_docstring(self):
        def add(a: int, b: int) -> int:
            return a + b

        t = FunctionTool(name="add", fn=add)
        assert t.parameters["properties"]["a"]["description"] == "a"

    def test_parses_typed_google_docstring(self):
        def process(data: str, verbose: bool = False) -> str:
            """Process data.

            Args:
                data (str): The input data to process.
                verbose (bool): Enable verbose output.

            Returns:
                Processed result.
            """
            return data

        t = FunctionTool(name="process", fn=process)
        props = t.parameters["properties"]
        assert props["data"]["description"] == "The input data to process."
        assert props["verbose"]["description"] == "Enable verbose output."


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


# --- Replay-safety class tests ---


class TestReplayClass:
    """The ``replay_class`` field: strict validation + serialization round-trip.

    Default is the safe ``side_effecting``; values are never auto-inferred, and
    an out-of-set value raises ``ValueError`` loudly at construction.
    """

    def test_default_is_side_effecting(self):
        def fn() -> int:
            return 1

        assert FunctionTool(name="f", fn=fn).replay_class == "side_effecting"
        assert RESTTool(name="r", url="https://example.com").replay_class == "side_effecting"
        assert (
            MCPTool(name="m", server_url="http://localhost:3000").replay_class
            == "side_effecting"
        )

    def test_explicit_values_accepted(self):
        def fn() -> int:
            return 1

        assert (
            FunctionTool(name="f", fn=fn, replay_class="read_only").replay_class
            == "read_only"
        )
        assert (
            RESTTool(
                name="r", url="https://example.com", method="GET", replay_class="read_only"
            ).replay_class
            == "read_only"
        )
        assert (
            MCPTool(
                name="m", server_url="http://localhost:3000", replay_class="idempotent"
            ).replay_class
            == "idempotent"
        )

    @pytest.mark.parametrize("bad", ["readonly", "read-only", "SIDE_EFFECTING", "", "none"])
    def test_invalid_value_raises_value_error(self, bad):
        def fn() -> int:
            return 1

        with pytest.raises(ValueError):
            FunctionTool(name="f", fn=fn, replay_class=bad)
        with pytest.raises(ValueError):
            RESTTool(name="r", url="https://example.com", replay_class=bad)
        with pytest.raises(ValueError):
            MCPTool(name="m", server_url="http://localhost:3000", replay_class=bad)

    def test_decorator_accepts_and_validates(self):
        @tool(name="lookup", replay_class="read_only")
        def lookup(q: str) -> str:
            """Look something up."""
            return q

        assert lookup.replay_class == "read_only"

        with pytest.raises(ValueError):

            @tool(name="bad_tool", replay_class="nope")
            def bad_tool(q: str) -> str:
                return q

    def test_to_dict_includes_replay_class(self):
        def fn() -> int:
            return 1

        assert FunctionTool(name="f", fn=fn).to_dict()["replay_class"] == "side_effecting"
        assert (
            RESTTool(name="r", url="https://example.com", replay_class="read_only").to_dict()[
                "replay_class"
            ]
            == "read_only"
        )

    def test_rest_roundtrip_preserves_replay_class(self):
        rt = RESTTool(
            name="r", url="https://example.com", method="GET", replay_class="read_only"
        )
        rebuilt = Tool.from_dict(rt.to_dict())
        assert isinstance(rebuilt, RESTTool)
        assert rebuilt.replay_class == "read_only"

    def test_mcp_roundtrip_preserves_replay_class(self):
        m = MCPTool(name="m", server_url="http://localhost:3000", replay_class="idempotent")
        rebuilt = Tool.from_dict(m.to_dict())
        assert isinstance(rebuilt, MCPTool)
        assert rebuilt.replay_class == "idempotent"

    def test_function_roundtrip_preserves_replay_class(self):
        def fn() -> int:
            return 1

        ft = FunctionTool(name="f_rt", fn=fn, replay_class="read_only")
        rebuilt = Tool.from_dict(ft.to_dict())
        assert rebuilt.replay_class == "read_only"

    def test_from_dict_missing_replay_class_defaults_safe(self):
        # An older serialized tool without the key resolves to the safe default.
        rebuilt = Tool.from_dict(
            {"name": "r", "tool_type": "rest_api", "config": {"url": "https://example.com"}}
        )
        assert rebuilt.replay_class == "side_effecting"


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


# ---------------------------------------------------------------------------
# security_review_1.md H1 — RESTTool SSRF hardening
# ---------------------------------------------------------------------------


class TestRESTToolSSRF:
    """Regression tests for security_review_1.md H1.

    The previous RESTTool used ``httpx.AsyncClient(follow_redirects=True)``
    with no host validation. Two SSRF shapes are now blocked:

    * The configured ``url`` is rejected if the host is private, loopback,
      link-local (incl. cloud metadata), reserved or multicast.
    * For ``body_mapping == "path_params"`` the post-substitution URL must
      keep the same host as the template.
    """

    @pytest.mark.asyncio
    async def test_loopback_url_blocked(self):
        from fastaiagent._internal.errors import ToolExecutionError

        t = RESTTool(name="local", url="http://127.0.0.1:7842/api/auth/status")
        with pytest.raises(ToolExecutionError, match="non-public|public address"):
            await t.aexecute({})

    @pytest.mark.asyncio
    async def test_link_local_metadata_blocked(self):
        from fastaiagent._internal.errors import ToolExecutionError

        t = RESTTool(
            name="metadata",
            url="http://169.254.169.254/latest/meta-data/",
        )
        with pytest.raises(ToolExecutionError, match="non-public|public address"):
            await t.aexecute({})

    @pytest.mark.asyncio
    async def test_path_param_pivot_via_developer_bug_template_blocked(self):
        """If a developer puts a placeholder in the host segment by mistake,
        the netloc-equality check refuses the request before issuing it.

        ``url.replace("{key}", value)`` is a string substitution — if the
        template happens to put ``{var}`` inside the authority (a developer
        bug) the LLM controls the host. Even though the SSRF guard would
        also catch a private-IP target, we surface a clearer error here.
        """
        from fastaiagent._internal.errors import ToolExecutionError

        t = RESTTool(
            name="lookup",
            # Placeholder in the host — a developer bug we still defend.
            url="https://{host}/api",
            body_mapping="path_params",
        )
        with pytest.raises(
            ToolExecutionError, match="changed the host|non-public|public address"
        ):
            await t.aexecute({"host": "evil.attacker"})

    @pytest.mark.asyncio
    async def test_private_allowed_with_env_opt_in(self, monkeypatch):
        """Intranet users opt in; the SSRF guard no longer trips."""
        from fastaiagent._internal.errors import ToolExecutionError
        from fastaiagent.multimodal._http import ALLOW_PRIVATE_NETWORKS_ENV

        monkeypatch.setenv(ALLOW_PRIVATE_NETWORKS_ENV, "1")
        t = RESTTool(name="dev", url="http://127.0.0.1:1/no-listener")
        with pytest.raises(ToolExecutionError) as excinfo:
            await t.aexecute({})
        # Some non-SSRF error (connection refused etc.) — not the guard.
        assert "non-public" not in str(excinfo.value).lower()
        assert "public address" not in str(excinfo.value).lower()
