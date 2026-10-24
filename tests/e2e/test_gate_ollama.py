"""End-to-end quality gate — Ollama provider against a real local daemon.

Skipped cleanly when Ollama is not running (CI runners, laptops without
``ollama serve``). When the daemon is reachable, exercises the full
``LLMClient(provider='ollama', ...)`` dispatch path:

- Direct LLMClient.complete()
- Agent.run() with the Ollama backend
- Phase A span instrumentation (root span, llm.ollama.* span)
- Replay reconstruction from a stored Ollama trace

Uses ``gemma2:2b-instruct-q8_0`` because it's small (~2.8GB), fast on
CPU, and supports the standard chat format. The model is auto-detected
from the running daemon's catalog so the gate can adapt if a different
small model is available.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from tests.e2e.conftest import require_ollama_running

pytestmark = pytest.mark.e2e


_PREFERRED_MODELS = [
    "gemma2:2b-instruct-q8_0",
    "gemma2:2b",
    "phi3:3.8b-mini-128k-instruct-q8_0",
    "phi3:mini",
    "llama3.1:8b-instruct-q6_K",
    "llama3.1:8b",
]


def _pick_available_model(host: str = "http://localhost:11434") -> str | None:
    """Return the first preferred model that's installed, or any small model."""
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None
    installed = {m.get("name", "") for m in data.get("models", [])}
    for preferred in _PREFERRED_MODELS:
        if preferred in installed:
            return preferred
    # Fallback: any installed model at all (test will run, but slowly).
    return next(iter(installed)) if installed else None


class TestOllamaProviderGate:
    """Real Ollama provider — daemon required, model auto-detected."""

    def test_01_llmclient_direct_complete(self, gate_state: dict[str, Any]) -> None:
        require_ollama_running()
        from fastaiagent import LLMClient
        from fastaiagent.llm.message import UserMessage

        model = _pick_available_model()
        if model is None:
            pytest.skip("Ollama running but no models installed — pull one with `ollama pull gemma2:2b`")
        gate_state["ollama_model"] = model

        llm = LLMClient(provider="ollama", model=model)
        response = llm.complete(
            [UserMessage("Reply with exactly the word: pong")]
        )
        assert response.content, "Ollama returned empty content"
        assert "pong" in response.content.lower(), (
            f"Ollama did not echo the expected word: {response.content!r}"
        )
        assert response.usage.get("total_tokens", 0) > 0, (
            "Ollama usage accounting missing total_tokens"
        )

    def test_02_agent_run_with_ollama(self, gate_state: dict[str, Any]) -> None:
        require_ollama_running()
        from fastaiagent import Agent, LLMClient

        model = gate_state.get("ollama_model") or _pick_available_model()
        if model is None:
            pytest.skip("no Ollama models installed")

        agent = Agent(
            name="ollama-gate",
            system_prompt=(
                "You are a terse assistant. Reply in one short sentence."
            ),
            llm=LLMClient(provider="ollama", model=model),
        )
        result = agent.run("Say hello.")
        assert result.output, "Ollama agent returned empty output"
        assert result.trace_id, "Ollama agent emitted no trace_id"
        assert result.tokens_used > 0, "Ollama agent token accounting broken"
        gate_state["ollama_trace_id"] = result.trace_id

    def test_03_replay_load_with_ollama_trace(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_ollama_running()
        from fastaiagent.trace.replay import Replay

        trace_id = gate_state.get("ollama_trace_id")
        if not trace_id:
            pytest.skip("test_02 did not produce a trace_id")

        replay = Replay.load(trace_id)
        steps = replay.steps()
        assert len(steps) >= 2, (
            f"expected >=2 spans (root agent + llm), got {len(steps)}: "
            f"{[s.span_name for s in steps]}"
        )
        # Phase A reconstruction attrs must be present
        root_attrs = replay._trace.spans[0].attributes
        for key in ("agent.config", "agent.tools", "agent.llm.config"):
            assert key in root_attrs, (
                f"Phase A reconstruction attr missing on Ollama trace: {key}"
            )
        assert root_attrs["agent.llm.provider"] == "ollama", (
            "agent.llm.provider did not round-trip as 'ollama'"
        )
        # The LLMClient.acomplete wrap from the previous PR should have
        # produced an llm.ollama.<model> span on every provider, including
        # Ollama, even though the integrations/openai monkey patch is
        # specific to OpenAI.
        llm_spans = [s for s in steps if s.span_name.startswith("llm.")]
        assert llm_spans, (
            "No llm.* spans for Ollama — LLMClient.acomplete wrap regressed"
        )
        assert any("ollama" in s.span_name for s in llm_spans), (
            f"llm.* span name does not include provider: "
            f"{[s.span_name for s in llm_spans]}"
        )


_TOOL_MODELS = ["llama3.1:8b-instruct-q6_K", "llama3.1:8b", "qwen3:8b", "mistral-nemo"]


def _pick_tool_model() -> str:
    """A model that calls tools (gemma2 and phi3 do not), or skip."""
    require_ollama_running()
    resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
    installed = {m.get("name", "") for m in resp.json().get("models", [])}
    for preferred in _TOOL_MODELS:
        match = next((name for name in installed if name.startswith(preferred)), None)
        if match:
            return match
    pytest.skip(f"no tool-capable Ollama model installed (pull one of {_TOOL_MODELS})")


def _secret_code(city: str) -> str:
    """The secret code for a city."""
    return "ZEBRA-4417" if "paris" in city.lower() else "NONE"


def _population(city: str) -> float:
    """The population of a city in millions."""
    return 2.1 if "lyon" in city.lower() else 0.0


def _tool_agent(**kwargs: Any) -> Any:
    from fastaiagent import Agent, LLMClient

    return Agent(
        name="ollama-tools",
        system_prompt=(
            "You have no knowledge of your own: always call a tool first, then "
            "report the tool's value exactly."
        ),
        # An 8B model on default sampling skips the tool now and then.
        llm=LLMClient(provider="ollama", model=_pick_tool_model(), temperature=0, seed=7),
        **kwargs,
    )


class TestOllamaToolLoop:
    """1.93.0: every Ollama tool loop failed on its second request (arguments
    sent as a JSON string, HTTP 400) and streamed tool calls were dropped."""

    def test_a_tool_loop_completes_on_run(self) -> None:
        from fastaiagent import FunctionTool

        agent = _tool_agent(tools=[FunctionTool(name="get_secret_code", fn=_secret_code)])
        result = agent.run("What is the secret code for Paris?")
        assert [tc["tool_name"] for tc in result.tool_calls] == ["get_secret_code"]
        assert "4417" in result.output

    def test_a_tool_loop_completes_on_astream(self) -> None:
        import asyncio

        from fastaiagent import FunctionTool
        from fastaiagent.llm.stream import TextDelta, ToolCallEnd

        agent = _tool_agent(tools=[FunctionTool(name="get_secret_code", fn=_secret_code)])

        async def main() -> tuple[list[str], str]:
            calls: list[str] = []
            text = ""
            async for event in agent.astream("What is the secret code for Paris?"):
                if isinstance(event, ToolCallEnd):
                    calls.append(event.tool_name)
                elif isinstance(event, TextDelta):
                    text += event.text
            return calls, text

        calls, text = asyncio.run(main())
        assert calls == ["get_secret_code"]
        assert "4417" in text

    def test_structured_output_beside_a_tool_uses_the_tool(self) -> None:
        from pydantic import BaseModel

        from fastaiagent import FunctionTool

        class CityFact(BaseModel):
            city: str
            population_millions: float

        agent = _tool_agent(
            tools=[FunctionTool(name="get_population", fn=_population)], output_type=CityFact
        )
        result = agent.run("Use get_population for Lyon and report it.")
        assert [tc["tool_name"] for tc in result.tool_calls] == ["get_population"]
        assert result.parsed is not None and result.parsed.population_millions == 2.1
