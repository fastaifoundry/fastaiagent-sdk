"""Unit tests for PersistentFactBlock — no LLM calls, no mocking."""

from __future__ import annotations

import pytest

from fastaiagent.agent.memory_blocks import PersistentFactBlock
from fastaiagent.learn import Fact, MemoryStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    from fastaiagent._internal.config import reset_config

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    reset_config()
    yield MemoryStore()
    reset_config()


def test_render_returns_empty_when_no_facts(store) -> None:
    block = PersistentFactBlock(scope="agent", scope_id="empty")
    assert block.render(query="anything") == []


def test_render_emits_system_message_with_facts(store) -> None:
    store.add(Fact(scope="agent", scope_id="x", fact="fact one"))
    store.add(Fact(scope="agent", scope_id="x", fact="fact two"))
    block = PersistentFactBlock(scope="agent", scope_id="x")
    rendered = block.render(query="any")
    assert len(rendered) == 1
    text = rendered[0].content
    assert "fact one" in text
    assert "fact two" in text
    assert "Learned facts" in text
    # scope label includes scope:scope_id
    assert "agent:x" in text


def test_on_message_is_no_op(store) -> None:
    """Block must not write to the store on conversational messages."""
    from fastaiagent.llm.message import UserMessage

    block = PersistentFactBlock(scope="agent", scope_id="x")
    block.on_message(UserMessage("just a message"))
    assert store.list_active(scope="agent", scope_id="x") == []


def test_max_facts_caps_injection(store) -> None:
    for i in range(10):
        store.add(Fact(scope="agent", scope_id="x", fact=f"fact-{i}"))
    block = PersistentFactBlock(scope="agent", scope_id="x", max_facts=3)
    rendered = block.render(query="any")
    text = rendered[0].content
    # Newest 3 — created last → highest index
    assert "fact-9" in text
    assert "fact-8" in text
    assert "fact-7" in text
    assert "fact-0" not in text


def test_invalid_scope_rejected() -> None:
    with pytest.raises(ValueError, match="scope"):
        PersistentFactBlock(scope="bad")  # type: ignore[arg-type]


def test_max_facts_positive_required() -> None:
    with pytest.raises(ValueError, match="max_facts"):
        PersistentFactBlock(scope="agent", max_facts=0)


def test_refresh_every_caches(store) -> None:
    """With refresh_every=N, the next DB refresh fires on render N+1."""
    store.add(Fact(scope="agent", scope_id="x", fact="alpha"))
    block = PersistentFactBlock(scope="agent", scope_id="x", refresh_every=5)

    # Render 1 — refresh fires (cache was None), counter starts at 1.
    rendered = block.render(query="any")
    assert "alpha" in rendered[0].content

    # Add a fact AFTER caching.
    store.add(Fact(scope="agent", scope_id="x", fact="beta"))

    # Renders 2..5 — cache hits, no refresh.
    for _ in range(4):
        rendered = block.render(query="any")
        assert "beta" not in rendered[0].content

    # Render 6 — counter has reached refresh_every (5), refresh fires.
    rendered = block.render(query="any")
    assert "beta" in rendered[0].content
