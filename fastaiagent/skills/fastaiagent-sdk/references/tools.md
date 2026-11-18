# Tools

Read this before defining or changing any tool: `@tool`, `FunctionTool`, `RESTTool`, `MCPTool`, the tool registry, deferred tools, `replay_class`, or a tool-call budget. Agent construction and middleware are in `agents.md`.

## Imports

- `from fastaiagent.tool import tool, FunctionTool, RESTTool, MCPTool, ToolRegistry, ToolResult, Tool` (all of them also import from `fastaiagent`).
- `fastaiagent.tool` is both the package and, at the top level, the decorator. Write `from fastaiagent.tool import X`; do not write `import fastaiagent.tool as t` and expect the module.
- `Agent(tools=[...])` takes `Tool` instances only. A plain function in that list fails; wrap it first.

## @tool and FunctionTool

```signature
fastaiagent.tool.function.tool(name=, description=, replay_class=, validate_args=, timeout=, max_retries=, retry_delay=, output_type=, register=, deferred=)
fastaiagent.FunctionTool(name, fn=, description=, parameters=, replay_class=, validate_args=, timeout=, max_retries=, retry_delay=, output_type=, register=, deferred=)
```

- `@tool(name="x")` and bare `@tool` both work. The result is a `FunctionTool`, not your function: `my_tool("a")` raises `TypeError`. Run it with `my_tool.execute({"arg": "a"})` (sync) or `await my_tool.aexecute(...)`; keep a separate plain function if you also need to call it directly.
- `FunctionTool(name=..., fn=...)`: `name` is required and is the name the model sees. With `@tool`, `name` defaults to the function name.
- The JSON schema is generated from type hints. Give every parameter a type hint: an unannotated parameter is typed `str`. Parameters without a default are `required`. Pydantic models, `Enum`, `Literal`, `Optional` produce a full schema.
- Parameter descriptions come from the docstring (`Args:` Google style, NumPy, or `:param x:`); the description is the docstring unless `description=` is given. Write a one-line docstring: it is what the model reads to decide when to call the tool. Pass `parameters=` (a JSON schema dict) to override generation.
- A parameter annotated `RunContext[T]` receives `Agent.run(context=RunContext(state=...))` and is hidden from the model. If the run has no `context=`, the call fails with a missing-argument error that is fed back to the model.
- `validate_args=True` (default) coerces the model's arguments to your hints (`"21"` becomes `21`). Invalid arguments return an error `ToolResult` the model sees; the function is not called.
- `async def` tools are supported. A sync tool with `timeout=` runs in a worker thread: on timeout the agent moves on, but the thread cannot be killed and finishes in the background.
- `timeout=` (seconds, must be > 0), `max_retries=` (>= 0) with exponential backoff `retry_delay * 2**attempt`, `output_type=` (validates/coerces the return value; a mismatch becomes an error result). Invalid values raise `ValueError` at construction. Only set `max_retries` on tools that are safe to repeat.
- A tool that raises does not abort the run: after retries the agent loop hands the model `Error: Tool '<name>' failed: ...`. Calling `.execute()` yourself raises `ToolExecutionError` (`fastaiagent._internal.errors`).
- `ToolResult(output=, error=, metadata=)`: `.success` is `error is None`. A custom tool may return `ToolResult(error="...")` instead of raising.
- Tool names must be unique inside one agent (middleware tools included, `ValueError`), and `final_result` is reserved when `output_mode="tool"`.

```python
from fastaiagent import Agent
from fastaiagent.testing import FunctionModel
from fastaiagent.tool import tool


@tool(name="add", replay_class="read_only")
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


assert add.execute({"a": "1", "b": 2}).output == 3  # "1" is coerced to 1


def responder(messages):
    if any(m.role.value == "tool" for m in messages):
        return "3"
    return "", [{"name": "add", "arguments": {"a": 1, "b": 2}}]


result = Agent(name="calc", llm=FunctionModel(responder), tools=[add]).run("1+2?")
assert result.output == "3"
```

```python
from dataclasses import dataclass

from fastaiagent import Agent, RunContext
from fastaiagent.testing import FunctionModel
from fastaiagent.tool import tool


@dataclass
class Deps:
    tenant: str


@tool(name="whoami", replay_class="read_only", timeout=5, max_retries=1)
def whoami(ctx: RunContext[Deps], greeting: str) -> str:
    """Greet the current tenant.

    Args:
        greeting: the greeting word to use
    """
    return f"{greeting} {ctx.state.tenant}"


assert "ctx" not in whoami.parameters["properties"]  # injected, hidden from the model


def responder(messages):
    tool_msgs = [m.content for m in messages if m.role.value == "tool"]
    if tool_msgs:
        return tool_msgs[-1]
    return "", [{"name": "whoami", "arguments": {"greeting": "hi"}}]


agent = Agent(name="ctx", llm=FunctionModel(responder), tools=[whoami])
assert agent.run("go", context=RunContext(state=Deps(tenant="acme"))).output == "hi acme"
```

## ToolRegistry

```signature
fastaiagent.ToolRegistry.register(tool)
fastaiagent.ToolRegistry.get(name)
fastaiagent.ToolRegistry.all()
fastaiagent.ToolRegistry.unregister(name)
fastaiagent.ToolRegistry.clear()
```

- Every `FunctionTool` with a function registers itself by name at construction (`register=True`). `Agent.from_dict` and replay use the registry to rebind a tool name to live code. It is process-wide and last write wins: two tools with the same name silently replace each other. Give tools distinct names.
- `register=False` keeps a tool out (use it for test doubles and stubs, otherwise they replace the real tool for every later rebuild).
- `ToolRegistry.clear()` is for tests only.

## RESTTool

```signature
fastaiagent.RESTTool(name, url, method=, headers=, body_mapping=, description=, parameters=, replay_class=, timeout=, max_retries=, retry_delay=, output_type=, deferred=)
```

- `method` defaults to `"GET"` and is upper-cased. `url` and `name` are required.
- Pass `parameters=` (JSON schema). Without it the schema is empty and the model has no arguments to send.
- `body_mapping` decides where the arguments go: `"json_body"` (the default) sends them as a JSON body, even for GET; `"query_params"` puts them in the query string; `"path_params"` replaces `{placeholders}` in the URL. Use `query_params` for GET, `json_body` for POST/PUT, `path_params` for `/orders/{order_id}`. Any other value sends nothing, without error.
- `path_params` refuses an argument that changes the host (raises `ToolExecutionError`).
- The response is parsed as JSON, else returned as text; `metadata` carries `status_code` and `url`. A non-2xx status raises `ToolExecutionError`, so the model sees `Error: ...`.
- The HTTP call has fixed limits (30 s, 5 redirects, 25 MiB). `timeout=` is the tool-level wait on top.
- Private, loopback and cloud-metadata addresses are refused. `FASTAIAGENT_ALLOW_PRIVATE_NETWORKS=1` allows intranet APIs.
- Header values are used at request time but serialized as `[redacted]` (`to_dict()`, traces, pushed definitions). A tool rebuilt from a trace or definition refuses to run with that placeholder (an error result), so the original tool must be supplied again. Read credentials from environment variables; never hard-code them in a header.
- GET does not mean read-only: set `replay_class` deliberately (below).

```python
import os

from fastaiagent import RESTTool

weather = RESTTool(
    name="get_forecast",
    description="Get the forecast for a city.",
    url="https://api.example.com/v1/forecast",
    method="GET",
    headers={"X-API-Key": os.environ.get("WEATHER_API_KEY", "")},
    body_mapping="query_params",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    replay_class="read_only",  # a pure lookup, marked on purpose
    timeout=20,
)
assert weather.method == "GET" and weather.to_dict()["config"]["headers"]["X-API-Key"] == "[redacted]"
```

## MCPTool (needs the `mcp` extra)

```bash
pip install "fastaiagent[mcp]"
```

`fastaiagent[mcp-server]` is the same dependency (`mcp>=1.24,<2`), used for `Agent.as_mcp_server`. Without it, calling an `MCPTool` raises `ImportError` with that install line.

```signature
fastaiagent.MCPTool(name, server_url=, tool_name=, auth_token=, description=, parameters=, replay_class=, command=, env=, timeout=, max_retries=, retry_delay=, output_type=, deferred=)
fastaiagent.MCPTool.from_server(server_url=, auth_token=, command=, env=, **tool_kwargs)
fastaiagent.MCPTool.afrom_server(server_url=, auth_token=, command=, env=, **tool_kwargs)
```

- Prefer `MCPTool.from_server(...)`: it connects, lists the server's tools and returns one ready `MCPTool` per tool (name, description, schema filled in). `afrom_server` is the async form; `from_server` also works inside a running loop.
- Pass `server_url=` (remote, streamable HTTP) or `command=[...]` (local subprocess over stdio), never both (`ValueError`). `env=` adds to a minimal environment for local servers.
- `**tool_kwargs` (`replay_class=`, `timeout=`, `max_retries=`, `deferred=`...) apply to every returned tool. Set `replay_class` here once, per server, after checking what its tools do; mixed servers need per-tool classification afterwards.
- Every call opens a session, calls, and closes it (a local server restarts per call). The handshake gives up after 30 s; a call waits `timeout`, else 300 s.
- A server `isError` reply becomes an error result; an unreachable or non-MCP server raises `ToolExecutionError`.
- `to_dict()` carries only `server_url` and `tool_name`. `auth_token`, `command` and `env` are never serialized, so a `command=` tool rebuilt from a payload cannot run. HTTP addresses pass the same private-network check as `RESTTool` (loopback allowed).

```python
from fastaiagent import Agent, LLMClient, MCPTool


def build_agent() -> Agent:
    # needs: the mcp extra and a reachable MCP server (and OPENAI_API_KEY for the model)
    tools = MCPTool.from_server(
        command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp/docs"],
        replay_class="read_only",  # true only because this server's tools only read
        timeout=30,
    )
    return Agent(name="files", llm=LLMClient(provider="openai", model="gpt-4o-mini"), tools=tools)
```

**An MCP server is optional infrastructure.** `from_server` connects when it is called, so an agent
builder that calls it at construction makes the whole app (every agent, every question) fail when
the server is down or not configured. Build the tool only when the server is configured, and
without it give the agent no tool and an instruction to say the lookup is unavailable (not to guess).
Do not make up a server address; take it from the user or the environment.

**Test it against a real server, not a fake.** Do not monkeypatch `from_server` or the tool's call:
that tests your wiring, never MCP. A local stdio server needs no port and no network. Put it next to
the tests (needs the `mcp` extra):

```python
# tests/rates_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("rates")


@mcp.tool()
def get_shipping_rates(zip_code: str, weight_kg: float) -> str:
    """Shipping rates for a destination zip code and a package weight in kg."""
    return f"standard: ${4.5 + weight_kg * 1.2:.2f} to {zip_code}"


if __name__ == "__main__":
    mcp.run()  # stdio
```

```python
# tests/test_rates.py
import sys
from pathlib import Path

from fastaiagent import MCPTool

SERVER = str(Path(__file__).parent / "rates_server.py")


def test_the_real_mcp_tool_is_found_and_called():
    tools = MCPTool.from_server(command=[sys.executable, SERVER], replay_class="read_only")
    assert [t.name for t in tools] == ["get_shipping_rates"]
    result = tools[0].execute({"zip_code": "94105", "weight_kg": 2})
    assert result.error is None
    assert "94105" in result.output
```

## Deferred tools

- `deferred=True` (on `@tool`, `FunctionTool`, `RESTTool`, `MCPTool`, or via `from_server(deferred=True)`) keeps a tool's definition out of the model request until needed. The model is offered one built-in tool, `load_tools(names)`, whose description lists each deferred tool by name and first docstring line; a loaded tool stays available for the rest of the run.
- Use it for large toolsets (an MCP server with dozens of tools). Write a good first docstring line: it is all the model sees before loading.
- Calling a deferred tool that is not loaded returns an error telling the model to call `load_tools`. `load_tools` disappears once everything is loaded. It is local behaviour: `deferred` is not in `to_dict()`.

## replay_class: set it on every tool

```signature
fastaiagent.trace.replay.ForkedReplay.with_tool_policy(policy=, allow=)
fastaiagent.trace.replay.ForkedReplay.with_tool_override(name, tool)
fastaiagent.trace.replay.ForkedReplay.with_tool_stub(name, returns)
```

Exact values (any other string raises `ValueError` at construction, including through `from_dict`):

| `replay_class` | Meaning | A replay engine that reads it |
|---|---|---|
| `"read_only"` | No side effects; safe to call again | May re-execute the tool |
| `"idempotent"` | Same arguments converge to the same state | Injects the recorded output |
| `"side_effecting"` | Writes, payments, email, anything observable | Injects the recorded output, never re-executes |

- Unset means `"side_effecting"` (the safe default). It is never inferred: a `GET` `RESTTool`, or a function named `get_...`, is not assumed `read_only`, because a GET can have effects (a trigger endpoint, a counter, a billed call) and guessing wrong would re-fire it during replay.
- Do set it explicitly on every tool, deliberately: `read_only` for pure lookups, `idempotent` for upserts keyed by an id, `side_effecting` for the rest. Don't leave a pure lookup unmarked just because the default is safe: a replay engine can then never re-run it and only replays the recorded output. Don't mark something `read_only` to make a replay "work": a wrong `read_only` re-fires real effects.
- The value is serialized in `to_dict()` and stamped on every tool span as `fastaiagent.tool.replay_class` (provider-run tools and unknown tool names get `side_effecting`). An external replay engine reading traces uses that attribute to choose inject versus execute.
- The SDK's own `ForkedReplay.rerun()` does not read it. By default it blocks every tool you did not supply or allow (policy `block`), whatever its class, and REST/MCP tools rebuilt from a trace run only when named in `allow=`. Choose per rerun:
  - `with_tool_policy("live", allow=["lookup_order"])` runs the registered function live (`block`, `error`, `live`).
  - `with_tool_override(name, tool_or_callable)` replaces a tool; `with_tool_stub(name, returns=...)` returns a canned value.

```python
from fastaiagent import Replay


def rerun_without_side_effects(trace_id: str, step: int):
    # needs: a trace in the local trace store
    forked = Replay.load(trace_id).fork_at(step)
    forked.with_tool_policy("block", allow=["lookup_order"])  # lookup_order is read_only
    forked.with_tool_stub("charge_card", returns="ok")
    return forked.rerun()
```

## Tool budgets and call limits

```signature
fastaiagent.ToolBudget(max_calls=, message=)
```

- `Agent(middleware=[ToolBudget(max_calls=5)])` caps tool invocations per run. On the call after the cap the run ends cooperatively: `result.stop_reason == "tool_budget"`, `result.output == message`, no exception. The count is checkpointed, so a resumed run keeps counting. `max_calls` must be >= 1.
- It counts calls, not model turns. For loop length use `AgentConfig(max_iterations=)`. For spend use `AgentConfig(max_total_tokens=, max_cost_usd=)` (see `agents.md`).

```signature
fastaiagent.check_tool_call(tool_name, tool_input, agent)
fastaiagent.acheck_tool_call(tool_name, tool_input, agent)
```

- `check_tool_call` is for tools you run yourself, outside `fa.Agent` (your own loop or another framework's tool). It returns a `ToolDecision` (`.decision` is `"allow"`, `"deny"` or `"require_approval"`, plus `.reason`) from the managed approval policy; it never pauses, you act on the verdict. `agent=` is a pushed agent's name or its id.
- It allows when the SDK is not connected or no policy matches, and denies when the policy check cannot be reached. `fa.Agent` applies this gate to its own tools automatically; do not call it from inside one.

## Other tool kinds

- `decision_tool(questions, llm=, name=, description=)` gives an agent the Decisions API as a `FunctionTool`.
- `web_search()`, `code_execution()`, `file_search(stores)` are provider-run tools (`ProviderTool`) executed by the model's provider. They need a provider that supports them, and a `tool_call` guardrail cannot see their arguments beforehand.
- Knowledge bases expose retrieval with `LocalKB.as_tool()`.
