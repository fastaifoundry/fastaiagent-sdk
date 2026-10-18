"""Tools loaded when the model needs them (1.91.0, Bet 1 piece 1G).

Every tool definition goes into every request, so a big toolset — an MCP server's
forty tools — costs context on every turn whether or not it is used. A tool built
with ``deferred=True`` is not sent. Instead the model is offered one built-in tool,
``load_tools(names)``, whose description lists each deferred tool by name and its
first line; loading adds those definitions for the rest of the run. The loaded set
is middleware state, so it is checkpointed and a resume keeps it. A loaded tool is
governed and guarded like any other.

``deferred`` is local: it is not part of ``Tool.to_dict()`` (no new payload key),
so a plane-distributed copy of the agent sends every tool, as before.

Real ``Agent``; the model is the repo's ``MockLLMClient``.
"""

from __future__ import annotations

from typing import Any

import pytest

from fastaiagent import Agent, FunctionTool, SQLiteCheckpointer
from fastaiagent.chain.interrupt import Resume, interrupt
from fastaiagent.guardrail import Guardrail, GuardrailPosition, GuardrailResult
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import ToolCall
from fastaiagent.tool import tool
from fastaiagent.tool.mcp import MCPTool
from fastaiagent.tool.rest import RESTTool
from tests.conftest import MockLLMClient


def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo:{text}"


def lookup_invoice(invoice_id: str) -> str:
    """Look up an invoice by id.

    Returns the amount and the due date.
    """
    return f"invoice {invoice_id}: $40, due 2026-11-01"


def lookup_shipment(order_id: str) -> str:
    """Track a shipment by order id."""
    return f"order {order_id}: in transit"


def _approve(text: str) -> str:
    """Needs approval."""
    interrupt(reason="approve?", context={"text": text})
    return f"approved:{text}"


def _call(name: str, cid: str, **args: Any) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=cid, name=name, arguments=args)],
        finish_reason="tool_calls",
    )


def _answer(text: str = "done") -> LLMResponse:
    return LLMResponse(content=text, finish_reason="stop")


def _tools() -> list[FunctionTool]:
    return [
        FunctionTool(name="echo", fn=echo),
        FunctionTool(name="lookup_invoice", fn=lookup_invoice, deferred=True),
        FunctionTool(name="lookup_shipment", fn=lookup_shipment, deferred=True),
    ]


def _offered(call: dict[str, Any]) -> list[str]:
    return [t["function"]["name"] for t in (call["tools"] or [])]


def _load_tools_def(call: dict[str, Any]) -> dict[str, Any]:
    return next(t["function"] for t in call["tools"] if t["function"]["name"] == "load_tools")


def _tool_results(call: dict[str, Any]) -> list[str]:
    return [m.content for m in call["messages"] if m.role.value == "tool"]


def test_deferred_tools_are_not_sent_until_loaded() -> None:
    llm = MockLLMClient(
        responses=[
            _call("load_tools", "c1", names=["lookup_invoice"]),
            _call("lookup_invoice", "c2", invoice_id="INV-9"),
            _answer("It is $40."),
        ]
    )
    result = Agent(name="d", llm=llm, tools=_tools()).run("What do I owe on INV-9?")

    assert result.output == "It is $40."
    assert _offered(llm._calls[0]) == ["echo", "load_tools"]
    described = _load_tools_def(llm._calls[0])["description"]
    assert "lookup_invoice: Look up an invoice by id." in described
    assert "lookup_shipment: Track a shipment by order id." in described
    assert "Returns the amount" not in described  # first line only
    assert _offered(llm._calls[1]) == ["echo", "lookup_invoice", "load_tools"]
    assert _tool_results(llm._calls[2])[-1] == "invoice INV-9: $40, due 2026-11-01"


@pytest.mark.asyncio
async def test_the_same_on_astream() -> None:
    llm = MockLLMClient(
        responses=[
            _call("load_tools", "c1", names=["lookup_shipment"]),
            _call("lookup_shipment", "c2", order_id="A-17"),
            _answer(),
        ]
    )
    [e async for e in Agent(name="d", llm=llm, tools=_tools()).astream("where is A-17?")]
    assert _offered(llm._calls[1]) == ["echo", "lookup_shipment", "load_tools"]
    assert _tool_results(llm._calls[2])[-1] == "order A-17: in transit"


def test_load_tools_goes_away_once_everything_is_loaded() -> None:
    llm = MockLLMClient(
        responses=[
            _call("load_tools", "c1", names=["lookup_invoice", "lookup_shipment"]),
            _answer(),
        ]
    )
    Agent(name="d", llm=llm, tools=_tools()).run("q")
    assert _offered(llm._calls[1]) == ["echo", "lookup_invoice", "lookup_shipment"]


def test_an_unknown_name_is_told_to_the_model() -> None:
    llm = MockLLMClient(responses=[_call("load_tools", "c1", names=["send_email"]), _answer()])
    Agent(name="d", llm=llm, tools=_tools()).run("q")
    (reply,) = _tool_results(llm._calls[1])
    assert "send_email" in reply and "lookup_invoice" in reply and "lookup_shipment" in reply
    assert _offered(llm._calls[1]) == ["echo", "load_tools"]


def test_a_deferred_tool_called_without_loading_is_not_run() -> None:
    ran: list[str] = []

    def secret(x: str) -> str:
        """Do the secret thing."""
        ran.append(x)
        return "ran"

    llm = MockLLMClient(responses=[_call("secret", "c1", x="1"), _answer()])
    Agent(name="d", llm=llm, tools=[FunctionTool(name="secret", fn=secret, deferred=True)]).run("q")
    assert ran == []
    assert "load_tools" in _tool_results(llm._calls[1])[0]


def test_a_loaded_tool_is_guarded_like_any_other() -> None:
    judged: list[str] = []

    def record(text: str) -> GuardrailResult:
        judged.append(text)
        return GuardrailResult(passed=True)

    llm = MockLLMClient(
        responses=[
            _call("load_tools", "c1", names=["lookup_invoice"]),
            _call("lookup_invoice", "c2", invoice_id="INV-9"),
            _answer(),
        ]
    )
    Agent(
        name="d",
        llm=llm,
        tools=_tools(),
        guardrails=[Guardrail(name="rec", position=GuardrailPosition.tool_call, fn=record)],
    ).run("q")
    assert any("INV-9" in j for j in judged)


def test_the_loaded_set_survives_a_resume(tmp_path) -> None:
    store = SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))
    llm = MockLLMClient(
        responses=[
            _call("load_tools", "c1", names=["lookup_invoice"]),
            _call("approve", "c2", text="ok?"),
            _call("lookup_invoice", "c3", invoice_id="INV-9"),
            _answer(),
        ]
    )
    agent = Agent(
        name="d",
        llm=llm,
        tools=[*_tools(), FunctionTool(name="approve", fn=_approve)],
        checkpointer=store,
    )
    assert agent.run("q", execution_id="dr").status == "paused"
    done = agent.resume("dr", resume_value=Resume(approved=True))
    assert done.status == "completed"
    after_resume = llm._calls[2]
    assert "lookup_invoice" in _offered(after_resume)
    assert _tool_results(llm._calls[3])[-1] == "invoice INV-9: $40, due 2026-11-01"


def test_no_deferred_tools_means_no_load_tools_and_parallel_stays_on() -> None:
    agent = Agent(name="d", tools=[FunctionTool(name="echo", fn=echo)])
    assert not agent._mw_pipeline  # nothing added: parallel tool calls stay allowed
    llm = MockLLMClient(responses=[_answer()])
    Agent(name="d", llm=llm, tools=[FunctionTool(name="echo", fn=echo)]).run("q")
    assert _offered(llm._calls[0]) == ["echo"]


def test_every_tool_kind_takes_deferred_and_it_is_not_serialized() -> None:
    @tool(deferred=True)
    def ping() -> str:
        """Ping."""
        return "pong"

    kinds = [
        ping,
        FunctionTool(name="f", fn=echo, deferred=True),
        MCPTool(name="m", server_url="http://localhost:9/mcp", deferred=True),
        RESTTool(name="r", url="http://localhost:9/x", deferred=True),
    ]
    assert all(t.deferred for t in kinds)
    assert all("deferred" not in t.to_dict() for t in kinds)
    assert FunctionTool(name="g", fn=echo).deferred is False


@pytest.mark.asyncio
async def test_an_mcp_client_calls_a_deferred_tool_directly() -> None:
    """Deferral hides definitions from a MODEL. An MCP client calling an exposed
    tool by name has its own listing and nothing to load."""
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session

    from fastaiagent.tool.mcp_server import FastAIAgentMCPServer

    agent = Agent(name="d", llm=MockLLMClient(), tools=_tools())
    server = FastAIAgentMCPServer(agent, expose_tools=True)
    async with create_connected_server_and_client_session(server._server) as client:
        result = await client.call_tool("lookup_invoice", arguments={"invoice_id": "INV-9"})
    text = " ".join(c.text for c in result.content if hasattr(c, "text"))
    assert not result.isError
    assert "invoice INV-9: $40" in text


def test_an_optimize_candidate_keeps_the_deferred_loader() -> None:
    from fastaiagent.optimize.candidate import Candidate, apply_candidate

    base = Agent(name="d", llm=MockLLMClient(), tools=_tools())
    clone = apply_candidate(base, Candidate(system_prompt="Be brief."))
    assert any(t.name == "load_tools" for t in clone._run_tools())


def test_load_tools_is_a_reserved_name_when_tools_are_deferred() -> None:
    with pytest.raises(ValueError, match="load_tools"):
        Agent(
            name="d",
            tools=[
                FunctionTool(name="load_tools", fn=echo),
                FunctionTool(name="lookup_invoice", fn=lookup_invoice, deferred=True),
            ],
        )
