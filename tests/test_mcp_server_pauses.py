"""An agent served over MCP never hands a client a pause as its answer (1.78.0).

Since 1.74.0 ``arun()`` returns a pause (``status="paused"``, ``output=""``) when a
managed approval policy or an ``interrupt()`` stops a tool call. The MCP server
passed that ``""`` to the client as a *successful* answer, a paused Chain as the
text ``"null"``, and an agent with no checkpointer as a bare reason string. MCP
has no way to resume, so the only honest answer is an error that says the run is
paused and names it.

``expose_tools=True`` had a second hole: an inner tool called by name ran
through ``tool.aexecute`` directly, so the governance gate the agent loop applies
was never consulted. A tool the plane denies, or holds for approval, just ran.

Real MCP protocol (the upstream in-memory transport + ``ClientSession``), real
``Agent`` / ``Chain`` / ``SQLiteCheckpointer``, and the HTTP stand-in for the
plane's governance endpoints (``tests/_governance_plane.py``). The model is
scripted: what matters is whether the tool ran and what the client was told.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

mcp = pytest.importorskip("mcp")

from fastaiagent import Agent, Chain, FunctionTool, interrupt  # noqa: E402
from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer  # noqa: E402
from fastaiagent.llm.client import LLMResponse  # noqa: E402
from fastaiagent.llm.message import ToolCall  # noqa: E402
from fastaiagent.tool.mcp_server import FastAIAgentMCPServer  # noqa: E402
from tests._governance_plane import AGENT_ID, GovPlane, reset_connection, serve  # noqa: E402
from tests.conftest import MockLLMClient  # noqa: E402


def _calls(tool: str, args: dict[str, Any]) -> MockLLMClient:
    """A model that calls ``tool`` once, then answers from its result."""
    return MockLLMClient(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="call_1", name=tool, arguments=args)],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="done", finish_reason="stop"),
        ]
    )


def _bank() -> tuple[list[dict[str, Any]], FunctionTool]:
    ran: list[dict[str, Any]] = []

    def transfer_funds(amount: int, to: str) -> str:
        ran.append({"amount": amount, "to": to})
        return f"Transferred ${amount} to {to}."

    return ran, FunctionTool(
        name="transfer_funds",
        fn=transfer_funds,
        parameters={
            "type": "object",
            "properties": {"amount": {"type": "integer"}, "to": {"type": "string"}},
            "required": ["amount", "to"],
        },
    )


def _asks_manager(amount: int) -> str:
    interrupt("manager_approval", {"amount": amount})
    return f"refunded {amount}"


REFUND = FunctionTool(name="refund", fn=_asks_manager)


def _text(result: Any) -> str:
    return " ".join(c.text for c in result.content if hasattr(c, "text"))


@pytest.fixture
def plane(isolated_local_db: Path) -> Iterator[GovPlane]:
    import fastaiagent

    with serve() as (state, url):
        fastaiagent.connect(api_key="fa_k_gov_test", target=url)
        try:
            yield state
        finally:
            reset_connection()


async def _call(server: FastAIAgentMCPServer, name: str, args: dict[str, Any]) -> Any:
    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(server._server) as client:
        return await client.call_tool(name, arguments=args)


# --- the primary tool: a paused run is an error, never an answer -----------------


@pytest.mark.asyncio
async def test_a_policy_pause_is_an_error_naming_the_paused_run(
    plane: GovPlane, tmp_path: Path
) -> None:
    ran, tool = _bank()
    agent = Agent(
        name="banker",
        agent_id=AGENT_ID,
        llm=_calls("transfer_funds", {"amount": 500, "to": "Bob"}),
        tools=[tool],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )

    result = await _call(FastAIAgentMCPServer(agent), "banker", {"input": "Pay Bob $500."})

    # Before 1.78.0: isError False and an empty text — a successful non-answer.
    assert result.isError is True
    text = _text(result)
    assert "paused" in text and "policy_approval_required" in text
    assert "transfer_funds" in text
    # The run can be found and resolved by whoever operates the server.
    run_id = plane.pending_posts[0]["run_id"]
    assert run_id in text
    assert ran == []
    # The arguments stay out of the message — MCP clients are not the approver.
    assert "Bob" not in text


@pytest.mark.asyncio
async def test_an_interrupt_is_an_error_naming_the_paused_run(tmp_path: Path) -> None:
    agent = Agent(
        name="refunds",
        llm=_calls("refund", {"amount": 40}),
        tools=[REFUND],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )
    result = await _call(FastAIAgentMCPServer(agent), "refunds", {"input": "refund 40"})

    assert result.isError is True
    text = _text(result)
    assert "manager_approval" in text and "execution_id=" in text


@pytest.mark.asyncio
async def test_a_pause_with_no_checkpointer_says_it_could_not_be_saved() -> None:
    agent = Agent(name="refunds", llm=_calls("refund", {"amount": 40}), tools=[REFUND])
    result = await _call(FastAIAgentMCPServer(agent), "refunds", {"input": "refund 40"})

    # Before 1.78.0 the bare InterruptSignal reached the client as "manager_approval".
    assert result.isError is True
    text = _text(result)
    assert "manager_approval" in text and "checkpointer" in text


@pytest.mark.asyncio
async def test_a_paused_chain_is_an_error_not_null(tmp_path: Path) -> None:
    chain = Chain("refund-flow", checkpointer=SQLiteCheckpointer(str(tmp_path / "c.db")))
    chain.add_node(
        "refund",
        tool=REFUND,
        input_mapping={"amount": "40"},
    )
    result = await _call(chain.as_mcp_server(), "refund_flow", {"input": "refund please"})

    # Before 1.78.0: isError False and the text "null".
    assert result.isError is True
    assert "manager_approval" in _text(result)


@pytest.mark.asyncio
async def test_a_finished_run_still_answers(plane: GovPlane) -> None:
    agent = Agent(
        name="helper", llm=MockLLMClient([LLMResponse(content="hi", finish_reason="stop")])
    )
    result = await _call(FastAIAgentMCPServer(agent), "helper", {"input": "hello"})
    assert result.isError is False
    assert _text(result) == "hi"


# --- expose_tools: an inner tool called by name is governed ----------------------


def _governed_server(tmp_path: Path) -> tuple[list[dict[str, Any]], FastAIAgentMCPServer]:
    ran, tool = _bank()
    agent = Agent(
        name="banker",
        agent_id=AGENT_ID,
        llm=MockLLMClient([LLMResponse(content="unused", finish_reason="stop")]),
        tools=[tool],
    )
    return ran, FastAIAgentMCPServer(agent, expose_tools=True)


@pytest.mark.asyncio
async def test_an_inner_tool_that_needs_approval_is_refused(
    plane: GovPlane, tmp_path: Path
) -> None:
    ran, server = _governed_server(tmp_path)
    result = await _call(server, "transfer_funds", {"amount": 500, "to": "Bob"})

    # Before 1.78.0 the transfer ran: no decide call, no refusal.
    assert ran == []
    assert result.isError is True
    assert "requires approval" in _text(result)
    assert [c["tool_name"] for c in plane.decide_calls] == ["transfer_funds"]
    # Nothing could ever resolve a pause here, so none is registered.
    assert plane.pending_posts == []


@pytest.mark.asyncio
async def test_an_inner_tool_the_policy_denies_is_refused(plane: GovPlane, tmp_path: Path) -> None:
    plane.gated_decision = "deny"
    ran, server = _governed_server(tmp_path)
    result = await _call(server, "transfer_funds", {"amount": 500, "to": "Bob"})

    assert ran == []
    assert result.isError is True
    assert "transfers are blocked for this agent" in _text(result)


@pytest.mark.asyncio
async def test_an_inner_tool_is_refused_when_governance_cannot_answer(
    plane: GovPlane, tmp_path: Path
) -> None:
    plane.gated_decision = "error"
    ran, server = _governed_server(tmp_path)
    result = await _call(server, "transfer_funds", {"amount": 500, "to": "Bob"})

    assert ran == []
    assert result.isError is True
    assert "governance check unavailable" in _text(result)


@pytest.mark.asyncio
async def test_an_inner_tool_the_policy_allows_runs(plane: GovPlane, tmp_path: Path) -> None:
    plane.gated_decision = "allow"
    ran, server = _governed_server(tmp_path)
    result = await _call(server, "transfer_funds", {"amount": 5, "to": "Ann"})

    assert ran == [{"amount": 5, "to": "Ann"}]
    assert result.isError is False
    assert _text(result) == "Transferred $5 to Ann."


@pytest.mark.asyncio
async def test_an_ungoverned_agent_s_inner_tool_runs_without_asking(plane: GovPlane) -> None:
    ran, tool = _bank()
    agent = Agent(
        name="banker",  # no agent_id: not enrolled, so there is nothing to ask
        llm=MockLLMClient([LLMResponse(content="unused", finish_reason="stop")]),
        tools=[tool],
    )
    result = await _call(
        FastAIAgentMCPServer(agent, expose_tools=True),
        "transfer_funds",
        {"amount": 5, "to": "Ann"},
    )
    assert ran == [{"amount": 5, "to": "Ann"}]
    assert result.isError is False
    assert plane.decide_calls == []
