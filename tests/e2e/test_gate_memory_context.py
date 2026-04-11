"""End-to-end quality gate — AgentMemory, RunContext, dynamic prompts.

Three orthogonal features that all bypass the "fresh agent per turn"
pattern the main gate exercises:

- ``AgentMemory`` lets an agent remember previous turns in a conversation.
- ``RunContext`` injects typed runtime state (DB, user session, etc.)
  into tools without polluting the tool signature visible to the LLM.
- Callable ``system_prompt`` lets the prompt depend on the context at
  run-time, so different users can see different instructions from the
  same agent object.

Each one is a real path a developer will hit. Untested here, untested
anywhere.

**Known SDK subtlety surfaced by this gate**
    With ``from __future__ import annotations`` enabled, all parameter
    annotations on tool functions become strings that must be resolvable
    via ``get_type_hints(fn)`` against the function's ``__globals__``
    (i.e. the defining module's top-level namespace). If ``RunContext``
    is imported lazily inside a function scope, resolution fails
    silently and ``FunctionTool`` falls back to treating the context
    parameter as a plain string — the LLM then sees it in the JSON
    schema and hallucinates a string value, breaking context injection.
    The fix from a user's perspective is to import ``RunContext`` (and
    any dataclass referenced in tool type hints) at module level. That
    is what this test file does — see the module-level imports below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

# IMPORTANT: these two imports must be at module level (not inside the
# test methods) so that get_type_hints() on tool functions defined
# inside tests can resolve RunContext[AppState]. See the module docstring.
from fastaiagent.agent.context import RunContext  # noqa: E402

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


@dataclass
class AppState:
    """Minimal typed state for RunContext injection."""

    user_id: str
    tier: str
    greeting: str


class TestMemoryContextGate:
    """Memory, RunContext, and dynamic system_prompt — three-in-one gate."""

    def test_01_agent_memory_retains_prior_turns(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        from fastaiagent import Agent, LLMClient
        from fastaiagent.agent.memory import AgentMemory

        memory = AgentMemory()
        agent = Agent(
            name="memory-gate",
            system_prompt=(
                "You are a helpful assistant. Remember what the user tells you "
                "and use it in later responses."
            ),
            llm=LLMClient(provider="openai", model="gpt-4.1"),
            memory=memory,
        )

        r1 = agent.run("My favorite color is cerulean. Acknowledge it briefly.")
        assert r1.output, "first run returned empty output"
        assert len(memory) >= 2, (
            f"memory did not grow after first turn — expected user+assistant, "
            f"got {len(memory)} messages"
        )

        r2 = agent.run("What is my favorite color?")
        assert r2.output, "second run returned empty output"
        assert "cerulean" in r2.output.lower(), (
            f"agent did not recall the color from memory. "
            f"Response: {r2.output!r}"
        )
        gate_state["memory_size"] = len(memory)

    def test_02_run_context_injects_typed_state_into_tool(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        from fastaiagent import Agent, FunctionTool, LLMClient

        def whoami(ctx: RunContext[AppState]) -> str:
            """Return the current user's id and tier."""
            state = ctx.state
            return f"user_id={state.user_id} tier={state.tier}"

        agent = Agent(
            name="context-gate",
            system_prompt=(
                "You are a helpful assistant. You MUST call the `whoami` tool "
                "before answering any question about the current user or their "
                "account. Do not refuse or speculate — call the tool first, "
                "then report its exact output verbatim in your answer."
            ),
            llm=LLMClient(provider="openai", model="gpt-4.1"),
            tools=[FunctionTool(name="whoami", fn=whoami)],
        )

        ctx = RunContext(
            state=AppState(user_id="u-42", tier="platinum", greeting="howdy")
        )
        result = agent.run(
            "Call whoami and tell me what it returns — include both the user_id "
            "and the tier in your answer.",
            context=ctx,
        )
        assert result.output, "context-injected run returned empty output"
        lower = result.output.lower()
        assert "u-42" in lower or "42" in lower, (
            f"agent did not expose the context-injected user_id: {result.output!r}"
        )
        assert "platinum" in lower, (
            f"agent did not expose the context-injected tier: {result.output!r}"
        )
        # Tool call record should NOT have received ctx as an LLM argument —
        # context injection is invisible to the model.
        assert result.tool_calls, "whoami tool was not invoked"
        tc = result.tool_calls[0]
        assert tc["tool_name"] == "whoami"
        assert "ctx" not in tc.get("arguments", {}), (
            "RunContext parameter leaked into LLM-visible tool arguments"
        )

    def test_03_dynamic_system_prompt_uses_context(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        from fastaiagent import Agent, LLMClient

        def make_prompt(ctx: RunContext[AppState] | None) -> str:
            if ctx is None:
                return "You are a helpful assistant."
            return (
                f"You are a helpful assistant. Always start your reply with "
                f"the word {ctx.state.greeting!r} (including the quotes)."
            )

        agent = Agent(
            name="dynamic-prompt-gate",
            system_prompt=make_prompt,
            llm=LLMClient(provider="openai", model="gpt-4.1"),
        )

        ctx = RunContext(
            state=AppState(user_id="u-1", tier="free", greeting="howdy")
        )
        result = agent.run("Tell me something nice.", context=ctx)
        assert result.output, "dynamic-prompt agent returned empty output"
        assert "howdy" in result.output.lower(), (
            f"agent did not honor dynamic system prompt — "
            f"greeting {ctx.state.greeting!r} not in response: {result.output!r}"
        )
