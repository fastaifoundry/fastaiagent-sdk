"""Example 70: Mark tools with a replay-safety class (``replay_class``).

Every tool — and the ``@tool`` decorator — takes an optional ``replay_class``
that tells Agent Replay (and the central Replay engine) whether a recorded tool
call may be **re-executed** on a rerun or must have its **recorded output
injected** instead:

* ``"read_only"``      — no side effects; safe to call again (a GET, a pure lookup).
* ``"idempotent"``     — repeating with the same args converges; output is injected.
* ``"side_effecting"`` — writes / payments / emails; output is injected, never rerun.

The default is the safe ``"side_effecting"``. Marks are **explicit only** — a GET
``RESTTool`` is NOT auto-classified ``read_only`` (auto-inferring a re-executable
class would violate the replay-safety invariant). The resolved value is emitted
on every tool-call span as ``fastaiagent.tool.replay_class``.

Runnable as pytest (no API keys, no network):
    pytest examples/70_tool_replay_class.py -v
"""

from __future__ import annotations

import pytest

from fastaiagent import RESTTool
from fastaiagent.tool import FunctionTool, MCPTool, Tool, tool

# --- Hand-marked tools (never auto-inferred) ---------------------------------


def test_read_only_rest_tool() -> None:
    # A GET endpoint the developer KNOWS is a pure read — mark it read_only so
    # replay may re-execute it instead of injecting a possibly-stale body.
    forecast = RESTTool(
        name="get_forecast",
        description="Read today's forecast for a city.",
        url="https://api.example.com/forecast",
        method="GET",
        replay_class="read_only",
    )
    assert forecast.replay_class == "read_only"


def test_idempotent_function_tool() -> None:
    @tool(name="upsert_user", replay_class="idempotent")
    def upsert_user(user_id: str, name: str) -> str:
        """Create-or-update a user (same args converge to the same state)."""
        return f"user {user_id} set to {name}"

    assert upsert_user.replay_class == "idempotent"


def test_side_effecting_is_the_default() -> None:
    # Unmarked → the safe default. Replay injects the recorded output rather
    # than charging the card a second time.
    @tool(name="charge_card")
    def charge_card(amount_cents: int) -> str:
        """Charge the customer's card (a real side effect)."""
        return f"charged {amount_cents}"

    assert charge_card.replay_class == "side_effecting"


# --- Strict validation: a typo fails loudly at construction ------------------


def test_invalid_value_raises() -> None:
    with pytest.raises(ValueError):
        FunctionTool(name="x", fn=lambda: None, replay_class="readonly")


# --- The mark survives serialization round-trips -----------------------------


def test_replay_class_round_trips() -> None:
    mcp = MCPTool(
        name="search", server_url="http://localhost:3000", replay_class="read_only"
    )
    data = mcp.to_dict()
    assert data["replay_class"] == "read_only"
    assert Tool.from_dict(data).replay_class == "read_only"
