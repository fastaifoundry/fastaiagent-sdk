"""Live: no real or rejected key reaches local.db or an exception message (1.89.0).

The offline suites (``test_secret_serialization_sweep.py``,
``test_provider_error_scrub.py``) script the providers' error bodies. This file
checks the same promises against the real providers, including the one shape
the scrubber cannot know in advance: what each provider actually echoes back
for a rejected key.

Keys are never printed. Every check computes a boolean first and asserts on
that, so a failure cannot render the key into pytest's assertion output.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any

import pytest

pytestmark = pytest.mark.e2e

#: Bogus keys with distinctive tails. A provider that echoes a key half-masked
#: keeps the last few characters, so the tail is what must not survive.
BOGUS_OPENAI = "sk-proj-" + "a1B2c3D4e5F6g7H8i9J0" * 2 + "QZXW"
BOGUS_ANTHROPIC = "sk-ant-api03-" + "a1B2c3D4e5F6g7H8i9J0" * 2 + "QZXW"
BOGUS_GEMINI = "AIza" + "a1B2c3D4e5F6g7H8i9J0a1B2c3D4e5F" + "QZXW"


def _need(var: str) -> str:
    value = os.environ.get(var)
    if not value:
        pytest.skip(f"{var} not set")
    return value


@pytest.fixture
def traced_db(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    monkeypatch.delenv("FASTAIAGENT_API_KEY", raising=False)
    monkeypatch.delenv("FASTAIAGENT_TARGET", raising=False)
    reset_config()
    _project.set_project_id("e2e-secrets")
    otel.reset()
    try:
        yield db_path
    finally:
        otel.reset()
        _project.reset_for_testing()
        reset_config()


def _rows(db_path: Any) -> str:
    conn = sqlite3.connect(str(db_path))
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return "\n".join(
            repr(row)
            for t in tables
            for row in conn.execute(f'SELECT * FROM "{t}"')  # noqa: S608 — names from sqlite_master
        )
    finally:
        conn.close()


def test_a_real_guarded_run_stores_no_key(traced_db: Any) -> None:
    import fastaiagent as fa
    from fastaiagent.guardrail.builtins import banned_topics
    from fastaiagent.llm import LLMClient

    real = _need("OPENAI_API_KEY")

    # The shape S1 leaked through is now refused, and the refusal does not echo it.
    with pytest.raises(ValueError) as exc:
        banned_topics(["weapons"], llm={"model": "gpt-4o-mini", "api_key": real})
    echoed = real in str(exc.value)
    assert not echoed, "the refusal echoed the key"

    # The supported shape: shareable fields in the rule, the key from the env.
    agent = fa.Agent(
        name="e2e-secrets",
        system_prompt="Answer in one short sentence.",
        llm=LLMClient(model="gpt-4o-mini", api_key=real),
        guardrails=[banned_topics(["weapons"], llm={"model": "gpt-4o-mini"})],
    )
    result = agent.run("Suggest a quick pasta dish.")
    assert result.output, "the agent returned nothing"

    rows = _rows(traced_db)
    assert "agent.guardrails" in rows, "the carrier attribute was not recorded"
    leaked = real[-8:] in rows
    assert not leaked, "local.db holds the real key"


def test_openai_rejected_key_is_not_echoed(traced_db: Any) -> None:
    import fastaiagent as fa
    from fastaiagent._internal.errors import LLMProviderError
    from fastaiagent.llm import LLMClient

    _need("OPENAI_API_KEY")  # same gate as the rest of the live suite
    agent = fa.Agent(
        name="e2e-bogus-openai",
        system_prompt="s",
        llm=LLMClient(model="gpt-4o-mini", api_key=BOGUS_OPENAI),
    )
    with pytest.raises(LLMProviderError) as exc:
        agent.run("hi")
    err = exc.value

    assert err.status_code == 401
    assert err.provider_code == "invalid_api_key"
    assert err.provider_message and "Incorrect API key provided" in err.provider_message, (
        "OpenAI's rejection text changed shape; re-check the masked-echo pattern"
    )
    in_message = "QZXW" in str(err)
    in_rows = "QZXW" in _rows(traced_db)
    assert not in_message, "the exception message still carries the key's tail"
    assert not in_rows, "local.db still carries the key's tail"


def test_anthropic_rejected_key_is_not_echoed() -> None:
    import asyncio

    from fastaiagent._internal.errors import LLMProviderError
    from fastaiagent.llm import LLMClient, UserMessage

    _need("ANTHROPIC_API_KEY")
    client = LLMClient(provider="anthropic", model="claude-haiku-4-5", api_key=BOGUS_ANTHROPIC)
    with pytest.raises(LLMProviderError) as exc:
        asyncio.run(client.acomplete([UserMessage("hi")]))
    assert exc.value.status_code == 401
    in_message = "QZXW" in str(exc.value)
    assert not in_message, "the exception message still carries the key's tail"


def test_gemini_authenticates_by_header() -> None:
    """The header is how the key now travels, so a real call has to succeed on it."""
    from fastaiagent.llm import LLMClient, UserMessage

    real = _need("GEMINI_API_KEY")
    client = LLMClient(provider="gemini", model="gemini-2.5-flash", api_key=real)
    response = client.complete([UserMessage("Reply with the single word: ok")])
    assert response.content


def test_gemini_rejected_key_is_not_echoed() -> None:
    import asyncio

    from fastaiagent._internal.errors import LLMProviderError
    from fastaiagent.llm import LLMClient, UserMessage

    _need("GEMINI_API_KEY")
    client = LLMClient(provider="gemini", model="gemini-2.5-flash", api_key=BOGUS_GEMINI)
    with pytest.raises(LLMProviderError) as exc:
        asyncio.run(client.acomplete([UserMessage("hi")]))
    assert exc.value.status_code in (400, 401, 403)
    in_message = "QZXW" in str(exc.value)
    assert not in_message, "the exception message still carries the key's tail"
