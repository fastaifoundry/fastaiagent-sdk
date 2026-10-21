"""Team-level middleware against real models (1.91.0, Bet 1 piece 1C).

``tests/test_container_middleware.py`` proves the wiring offline. This runs a real
Swarm handoff, a real Supervisor delegation and a real Chain under one team
middleware and checks it saw every hop, every delegate call and every node.

Run with the keys from ``~/.zshrc``::

    zsh -lc 'unset FASTAIAGENT_API_KEY FASTAIAGENT_TARGET; \
      FASTAIAGENT_LOCAL_DB=<scratch>/local.db \
      .venv/bin/python -m pytest tests/e2e/test_container_middleware_live.py -m e2e -v -rs'
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.e2e

MODELS = [
    pytest.param("openai", "gpt-5-mini", "OPENAI_API_KEY", id="gpt-5-mini"),
    pytest.param("gemini", "gemini-3.8-flash", "GEMINI_API_KEY", id="gemini-3.8-flash"),
    pytest.param("anthropic", "claude-haiku-4-5", "ANTHROPIC_API_KEY", id="claude-haiku-4-5"),
]


def _need(key: str) -> None:
    if not os.environ.get(key):
        pytest.skip(f"{key} not set")


def get_invoice(invoice_id: str) -> str:
    """Look up an invoice: amount and due date."""
    return {"INV-9": "amount $40.00, due 2026-11-01"}.get(invoice_id, "no such invoice")


def _audit(log: list[str]):
    from fastaiagent import AgentMiddleware

    class Audit(AgentMiddleware):
        name = "audit"

        async def before_run(self, ctx, input):
            log.append(f"run:{ctx.agent_name}")
            return input

        async def wrap_tool(self, ctx, tool, args, call_next):
            log.append(f"tool:{tool.name}")
            return await call_next(tool, args)

    return Audit()


@pytest.mark.parametrize(("provider", "model", "key"), MODELS)
def test_swarm_middleware_sees_every_real_hop(provider: str, model: str, key: str) -> None:
    _need(key)
    from fastaiagent import Agent, FunctionTool, LLMClient, Swarm

    llm = LLMClient(provider=provider, model=model)
    triage = Agent(
        name="triage",
        system_prompt="You route requests. For anything about invoices, hand off to billing.",
        llm=llm,
    )
    billing = Agent(
        name="billing",
        system_prompt="You answer invoice questions with the get_invoice tool. Be brief.",
        llm=llm,
        tools=[FunctionTool(name="get_invoice", fn=get_invoice)],
    )
    log: list[str] = []
    swarm = Swarm(
        name="desk",
        agents=[triage, billing],
        entrypoint="triage",
        handoffs={"triage": ["billing"], "billing": []},
        middleware=[_audit(log)],
    )
    result = swarm.run("How much is invoice INV-9?")
    assert "40" in result.output
    assert log[0] == "run:triage" and "run:billing" in log
    assert "tool:handoff_to_billing" in log and "tool:get_invoice" in log


def test_supervisor_middleware_sees_the_real_delegation() -> None:
    _need("OPENAI_API_KEY")
    from fastaiagent import Agent, FunctionTool, LLMClient, Supervisor, Worker

    worker = Agent(
        name="billing",
        system_prompt="Answer invoice questions with get_invoice. Be brief.",
        llm=LLMClient(provider="openai", model="gpt-5-mini"),
        tools=[FunctionTool(name="get_invoice", fn=get_invoice)],
    )
    log: list[str] = []
    sup = Supervisor(
        name="lead",
        llm=LLMClient(provider="openai", model="gpt-5-mini"),
        workers=[Worker(agent=worker, role="billing", description="invoice questions")],
        middleware=[_audit(log)],
    )
    result = sup.run("Ask billing how much invoice INV-9 is.")
    assert "40" in result.output
    assert log[0] == "run:lead"
    assert "tool:delegate_to_billing" in log
    assert "tool:get_invoice" not in log  # the worker keeps its own middleware (none)


def test_chain_middleware_wraps_a_real_agent_node_and_a_tool_node() -> None:
    _need("GEMINI_API_KEY")
    from fastaiagent import Agent, FunctionTool, LLMClient
    from fastaiagent.chain import Chain, NodeType

    def lookup(invoice_id: str) -> dict:
        """Fetch an invoice record."""
        return {"invoice": invoice_id, "detail": get_invoice(invoice_id)}

    log: list[str] = []
    chain = Chain("billing-flow", checkpoint_enabled=False, middleware=[_audit(log)])
    chain.add_node(
        "fetch",
        tool=FunctionTool(name="lookup", fn=lookup),
        type=NodeType.tool,
        input_mapping={"invoice_id": "{{state.invoice_id}}"},
    )
    chain.add_node(
        "explain",
        agent=Agent(
            name="explainer",
            system_prompt="Explain the invoice record you are given in one sentence.",
            llm=LLMClient(provider="gemini", model="gemini-3.8-flash"),
        ),
    )
    chain.connect("fetch", "explain")
    result = chain.execute(
        {"invoice_id": "INV-9", "input": "Explain invoice INV-9: $40, due 2026-11-01."}
    )
    assert result.status == "completed"
    assert log == ["tool:lookup", "run:explainer"]
