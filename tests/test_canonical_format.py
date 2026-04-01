"""Contract tests — verify SDK serialization matches canonical fixture files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastaiagent.agent import Agent
from fastaiagent.chain import Chain
from fastaiagent.chain.node import Edge, NodeConfig
from fastaiagent.guardrail import Guardrail
from fastaiagent.prompt import Prompt
from fastaiagent.tool import Tool
from fastaiagent.trace.storage import TraceData

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


# --- Agent fixtures ---


class TestAgentFixtures:
    def test_agent_simple_from_dict(self):
        data = _load_fixture("agent_simple.json")
        agent = Agent.from_dict(data)
        assert agent.name == data["name"]
        assert agent.system_prompt == data["system_prompt"]
        assert len(agent.tools) == len(data["tools"])

    def test_agent_simple_roundtrip(self):
        data = _load_fixture("agent_simple.json")
        agent = Agent.from_dict(data)
        output = agent.to_dict()
        assert output["name"] == data["name"]
        assert output["system_prompt"] == data["system_prompt"]
        assert output["agent_type"] == data["agent_type"]
        assert output["llm_endpoint"]["provider"] == data["llm_endpoint"]["provider"]
        assert output["llm_endpoint"]["model"] == data["llm_endpoint"]["model"]

    def test_agent_with_guardrails_from_dict(self):
        data = _load_fixture("agent_with_guardrails.json")
        agent = Agent.from_dict(data)
        assert agent.name == data["name"]
        assert len(agent.guardrails) == len(data["guardrails"])
        assert agent.config.max_iterations == data["config"]["max_iterations"]

    def test_agent_with_guardrails_roundtrip(self):
        data = _load_fixture("agent_with_guardrails.json")
        agent = Agent.from_dict(data)
        output = agent.to_dict()
        assert output["name"] == data["name"]
        assert len(output["guardrails"]) == len(data["guardrails"])
        assert output["config"]["temperature"] == data["config"]["temperature"]


# --- Chain fixtures ---


class TestChainFixtures:
    def test_chain_simple_from_dict(self):
        data = _load_fixture("chain_simple.json")
        chain = Chain.from_dict(data)
        assert chain.name == data["name"]
        assert len(chain.nodes) == len(data["nodes"])
        assert len(chain.edges) == len(data["edges"])

    def test_chain_simple_roundtrip(self):
        data = _load_fixture("chain_simple.json")
        chain = Chain.from_dict(data)
        output = chain.to_dict()
        assert output["name"] == data["name"]
        assert len(output["nodes"]) == len(data["nodes"])
        assert len(output["edges"]) == len(data["edges"])

    def test_chain_cyclic_preserves_cycle_config(self):
        data = _load_fixture("chain_cyclic.json")
        chain = Chain.from_dict(data)
        output = chain.to_dict()

        cyclic_edges_in = [e for e in data["edges"] if e.get("is_cyclic")]
        cyclic_edges_out = [e for e in output["edges"] if e.get("is_cyclic")]
        assert len(cyclic_edges_in) == len(cyclic_edges_out)

        for e_in, e_out in zip(cyclic_edges_in, cyclic_edges_out):
            assert e_out["cycle_config"]["max_iterations"] == e_in["cycle_config"]["max_iterations"]

    def test_chain_typed_state_preserves_schema(self):
        data = _load_fixture("chain_typed_state.json")
        chain = Chain.from_dict(data)
        output = chain.to_dict()
        assert output["state_schema"] == data["state_schema"]

    def test_chain_full_roundtrip(self):
        data = _load_fixture("chain_full.json")
        chain = Chain.from_dict(data)
        output = chain.to_dict()
        assert output["name"] == data["name"]
        assert len(output["nodes"]) == len(data["nodes"])
        assert len(output["edges"]) == len(data["edges"])
        assert output.get("state_schema") == data.get("state_schema")


# --- Tool fixtures ---


class TestToolFixtures:
    def test_tool_function_roundtrip(self):
        data = _load_fixture("tool_function.json")
        tool = Tool.from_dict(data)
        output = tool.to_dict()
        assert output["name"] == data["name"]
        assert output["tool_type"] == data["tool_type"]
        assert output["parameters"] == data["parameters"]

    def test_tool_rest_roundtrip(self):
        data = _load_fixture("tool_rest.json")
        tool = Tool.from_dict(data)
        output = tool.to_dict()
        assert output["name"] == data["name"]
        assert output["tool_type"] == data["tool_type"]
        assert output["config"]["url"] == data["config"]["url"]
        assert output["config"]["method"] == data["config"]["method"]

    def test_tool_mcp_roundtrip(self):
        data = _load_fixture("tool_mcp.json")
        tool = Tool.from_dict(data)
        output = tool.to_dict()
        assert output["name"] == data["name"]
        assert output["tool_type"] == data["tool_type"]
        assert output["config"]["server_url"] == data["config"]["server_url"]


# --- Prompt fixtures ---


class TestPromptFixtures:
    def test_prompt_simple_roundtrip(self):
        data = _load_fixture("prompt_simple.json")
        prompt = Prompt.from_dict(data)
        output = prompt.to_dict()
        assert output["name"] == data["name"]
        assert output["template"] == data["template"]
        assert sorted(output["variables"]) == sorted(data["variables"])

    def test_prompt_with_fragments_roundtrip(self):
        data = _load_fixture("prompt_with_fragments.json")
        prompt = Prompt.from_dict(data)
        output = prompt.to_dict()
        assert output["name"] == data["name"]
        assert output["template"] == data["template"]


# --- Guardrail fixtures ---


class TestGuardrailFixtures:
    def test_guardrail_code_roundtrip(self):
        data = _load_fixture("guardrail_code.json")
        g = Guardrail.from_dict(data)
        output = g.to_dict()
        assert output["name"] == data["name"]
        assert output["guardrail_type"] == data["guardrail_type"]
        assert output["blocking"] == data["blocking"]

    def test_guardrail_llm_judge_roundtrip(self):
        data = _load_fixture("guardrail_llm_judge.json")
        g = Guardrail.from_dict(data)
        output = g.to_dict()
        assert output["name"] == data["name"]
        assert output["guardrail_type"] == data["guardrail_type"]
        assert output["config"]["pass_value"] == data["config"]["pass_value"]

    def test_guardrail_regex_roundtrip(self):
        data = _load_fixture("guardrail_regex.json")
        g = Guardrail.from_dict(data)
        output = g.to_dict()
        assert output["name"] == data["name"]
        assert output["guardrail_type"] == data["guardrail_type"]
        assert output["config"]["pattern"] == data["config"]["pattern"]


# --- Trace fixtures ---


class TestTraceFixtures:
    def test_trace_agent_from_dict(self):
        data = _load_fixture("trace_agent.json")
        trace = TraceData.model_validate(data)
        assert trace.trace_id == data["trace_id"]
        assert len(trace.spans) == len(data["spans"])
        assert trace.spans[0].name == "agent.run"

    def test_trace_chain_from_dict(self):
        data = _load_fixture("trace_chain.json")
        trace = TraceData.model_validate(data)
        assert trace.trace_id == data["trace_id"]
        assert len(trace.spans) == len(data["spans"])


# --- VERSION file ---


class TestVersion:
    def test_fixture_version(self):
        version = (FIXTURES / "VERSION").read_text().strip()
        assert version == "1.0"
