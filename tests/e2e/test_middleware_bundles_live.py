"""Middleware bundles and deferred tools against real models (1.91.0, Bet 1 1B + 1G).

The offline suites prove the wiring with a recording model. What only a real
model can show: that it finds a deferred tool from the ``load_tools`` description,
loads it and then calls it; and that a middleware's own tool and instructions are
used as written.

Run with the keys from ``~/.zshrc``::

    zsh -lc 'unset FASTAIAGENT_API_KEY FASTAIAGENT_TARGET; \
      FASTAIAGENT_LOCAL_DB=<scratch>/local.db \
      .venv/bin/python -m pytest tests/e2e/test_middleware_bundles_live.py -m e2e -v -rs'
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
    """Look up an invoice: its amount and due date."""
    return {"INV-9": "amount $40.00, due 2026-11-01"}.get(invoice_id, "no such invoice")


def track_shipment(order_id: str) -> str:
    """Track a shipment by order id."""
    return f"order {order_id}: in transit"


def issue_refund(order_id: str) -> str:
    """Issue a refund for an order."""
    return f"refund REF-{order_id} issued"


@pytest.mark.parametrize(("provider", "model", "key"), MODELS)
def test_a_real_model_loads_a_deferred_tool_then_calls_it(
    provider: str, model: str, key: str
) -> None:
    _need(key)
    from fastaiagent import Agent, FunctionTool, LLMClient

    agent = Agent(
        name="billing",
        system_prompt="Answer billing questions with the tools. Be brief.",
        llm=LLMClient(provider=provider, model=model),
        tools=[
            FunctionTool(name="get_invoice", fn=get_invoice, deferred=True),
            FunctionTool(name="track_shipment", fn=track_shipment, deferred=True),
        ],
    )
    result = agent.run("How much is invoice INV-9 and when is it due?")
    names = [c["tool_name"] for c in result.tool_calls]
    assert names[0] == "load_tools", names
    assert "get_invoice" in names, names
    assert "track_shipment" not in names
    assert "40" in result.output and ("2026-11-01" in result.output or "November" in result.output)


@pytest.mark.parametrize(("provider", "model", "key"), MODELS)
def test_a_middlewares_tool_and_instructions_are_used(provider: str, model: str, key: str) -> None:
    _need(key)
    from fastaiagent import Agent, AgentMiddleware, FunctionTool, LLMClient

    class Refunds(AgentMiddleware):
        name = "refunds"
        tools = [FunctionTool(name="issue_refund", fn=issue_refund)]
        instructions = (
            "Refund policy: you may refund any order the customer names. Always "
            "use the issue_refund tool and quote the refund id it returns."
        )

    agent = Agent(
        name="support",
        system_prompt="You are a support agent. Be brief.",
        llm=LLMClient(provider=provider, model=model),
        middleware=[Refunds()],
    )
    result = agent.run("Please refund order A-17.")
    assert [c["tool_name"] for c in result.tool_calls] == ["issue_refund"]
    assert "REF-A-17" in result.output
