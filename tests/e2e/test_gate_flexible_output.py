"""End-to-end gate — flexible structured output (real OpenAI + Anthropic).

Covers the v1.42.0 structured-output work:
  - non-model output types (``list[Model]``) end to end,
  - OpenAI native *strict* Structured Outputs (validates the generated strict
    schema is actually accepted by the API — the thing unit tests can't prove).

Retry-on-validation-failure is exercised deterministically in
``tests/test_output_type_flexible.py``; here we confirm the happy path stays
green with retries enabled (the default).
"""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel

pytestmark = pytest.mark.e2e


def _require(provider_key: str) -> None:
    if os.environ.get(provider_key):
        return
    message = f"{provider_key} not set — skipping flexible-output gate step"
    if os.environ.get("E2E_REQUIRED") == "1":
        pytest.fail(message)
    pytest.skip(message)


class Country(BaseModel):
    name: str
    capital: str


class Address(BaseModel):
    street: str
    unit: str | None = None  # optional → exercises strict nullability


class Company(BaseModel):
    name: str
    address: Address
    employee_count: int


def test_openai_list_output() -> None:
    _require("OPENAI_API_KEY")
    from fastaiagent import Agent, LLMClient

    agent = Agent(
        name="geo-openai",
        system_prompt="You extract structured data.",
        llm=LLMClient(provider="openai", model="gpt-4o"),
        output_type=list[Country],
    )
    result = agent.run("List France, Japan, and Egypt with their capitals.")
    assert isinstance(result.parsed, list) and len(result.parsed) == 3
    assert all(isinstance(c, Country) and c.name and c.capital for c in result.parsed)


def test_anthropic_list_output() -> None:
    _require("ANTHROPIC_API_KEY")
    from fastaiagent import Agent, LLMClient

    agent = Agent(
        name="geo-anthropic",
        system_prompt="You extract structured data.",
        llm=LLMClient(provider="anthropic", model="claude-sonnet-4-6"),
        output_type=list[Country],
    )
    result = agent.run("List France, Japan, and Egypt with their capitals.")
    assert isinstance(result.parsed, list) and len(result.parsed) == 3
    assert all(isinstance(c, Country) for c in result.parsed)


def test_openai_strict_output() -> None:
    _require("OPENAI_API_KEY")
    from fastaiagent import Agent, AgentConfig, LLMClient

    agent = Agent(
        name="company-strict",
        system_prompt="You extract structured company data.",
        llm=LLMClient(provider="openai", model="gpt-4o"),
        output_type=Company,
        config=AgentConfig(strict_output=True),
    )
    # If the generated strict schema were invalid, the OpenAI API would 400 here.
    result = agent.run(
        "Acme Corp, at 42 Main St unit 5, employs 250 people."
    )
    assert isinstance(result.parsed, Company)
    assert result.parsed.name
    assert isinstance(result.parsed.address, Address)
    assert result.parsed.employee_count == 250


def test_openai_primitive_output() -> None:
    _require("OPENAI_API_KEY")
    from fastaiagent import Agent, LLMClient

    agent = Agent(
        name="counter",
        system_prompt="You answer with a single number.",
        llm=LLMClient(provider="openai", model="gpt-4o"),
        output_type=int,
    )
    result = agent.run("How many sides does a hexagon have?")
    assert result.parsed == 6


def test_llmclient_level_parsed() -> None:
    # Structured output straight off the raw LLMClient — no Agent.
    _require("OPENAI_API_KEY")
    from fastaiagent import LLMClient
    from fastaiagent.llm.message import UserMessage

    client = LLMClient(provider="openai", model="gpt-4o")
    resp = client.complete(
        [UserMessage("Give France and its capital.")],
        output_type=Country,
    )
    assert isinstance(resp.parsed, Country)
    assert resp.parsed.name and resp.parsed.capital
