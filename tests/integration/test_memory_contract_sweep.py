"""Memory's promises, held across every axis at once (1.82.0).

Each memory fix so far came with a test for the one path it changed, so bugs
kept surviving on the combinations no test crossed: ``astream`` but not
``run``, tracing off, a Postgres or Redis store, a per-user resolver, an
optimized agent. This sweep states each contract once and runs it across

    store    — SQLite (always), Postgres (PG_TEST_DSN), Redis (REDIS_TEST_URL)
    path     — ``run`` and ``astream``
    tracing  — on and off

so a new combination is checked the day it exists. A combination that can't
hold yet belongs here as a signed-off ``xfail``, never as a missing case.

No mocking: real stores, the SDK's offline ``TestModel`` / ``FunctionModel``,
real middleware and tools. Postgres and Redis follow the conformance suite's
gating (``tests/integration/conftest.py`` fills the variables when the dev
containers are up).
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from fastaiagent import Agent, FunctionTool, Memory, RedactPII, RunContext
from fastaiagent._internal.config import reset_config
from fastaiagent.learn import Fact
from fastaiagent.llm.message import MessageRole
from fastaiagent.testing import FunctionModel, TestModel
from fastaiagent.trace.otel import reset as reset_tracer

PG_DSN = os.environ.get("PG_TEST_DSN")
REDIS_URL = os.environ.get("REDIS_TEST_URL")


@dataclass
class _Session:
    user_id: str


# ── axes ─────────────────────────────────────────────────────────────────────


@pytest.fixture(params=["sqlite", "postgres", "redis"])
def store(request: pytest.FixtureRequest, tmp_path) -> Iterator[Any]:
    if request.param == "sqlite":
        from fastaiagent.learn import MemoryStore

        yield MemoryStore(db_path=str(tmp_path / "facts.db"))
    elif request.param == "postgres":
        if not PG_DSN:
            pytest.skip("PG_TEST_DSN not set")
        from fastaiagent.learn import PostgresFactStore

        pg = PostgresFactStore(PG_DSN)
        yield pg
        pg.close()
    else:
        if not REDIS_URL:
            pytest.skip("REDIS_TEST_URL not set")
        from fastaiagent.learn import RedisFactStore

        yield RedisFactStore(REDIS_URL, namespace=f"sweep{uuid.uuid4().hex[:8]}")


@pytest.fixture(params=["run", "astream"])
def path(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=["tracing-on", "tracing-off"])
def tracing(request: pytest.FixtureRequest, tmp_path, monkeypatch) -> Iterator[str]:
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    monkeypatch.setenv("FASTAIAGENT_TRACE_ENABLED", "1" if request.param == "tracing-on" else "0")
    reset_config()
    reset_tracer()
    yield request.param
    reset_tracer()
    reset_config()


def _uid(name: str) -> str:
    return f"{name}-{uuid.uuid4().hex[:8]}"  # shared servers: never collide


def _turn(agent: Agent, text: str, uid: str | None, path: str) -> None:
    context = RunContext(state=_Session(uid)) if uid else None
    if path == "run":
        agent.run(text, context=context)
        return

    async def drain() -> None:
        async for _ in agent.astream(text, context=context):
            pass

    asyncio.run(drain())


def _prompt(model: Any) -> str:
    return " ".join(
        m.content or "" for m in model.calls[-1]["messages"] if isinstance(m.content, str)
    )


def _per_user(store: Any, **kwargs: Any) -> Memory:
    return Memory(location=store, user_id=lambda ctx: ctx.state.user_id, **kwargs)


# ── contracts ────────────────────────────────────────────────────────────────


def test_one_users_conversation_never_reaches_another(store, path, tracing):
    alice, bob = _uid("alice"), _uid("bob")
    model = TestModel(response="noted")
    agent = Agent(name="a", llm=model, memory=_per_user(store))
    _turn(agent, "alice-secret-7731", alice, path)
    _turn(agent, "hello", bob, path)
    assert "alice-secret-7731" not in _prompt(model)
    _turn(agent, "hello", None, path)  # a caller with no user
    assert "alice-secret-7731" not in _prompt(model)
    _turn(agent, "again", alice, path)
    assert "alice-secret-7731" in _prompt(model)  # her own window still has it


def test_one_users_facts_never_reach_another(store, path, tracing):
    alice, bob = _uid("alice"), _uid("bob")
    store.add(Fact(scope="user", scope_id=alice, fact="alice-fact-4410"))
    model = TestModel(response="noted")
    agent = Agent(name="a", llm=model, memory=_per_user(store))
    _turn(agent, "hi", bob, path)
    assert "alice-fact-4410" not in _prompt(model)
    _turn(agent, "hi", alice, path)
    assert "alice-fact-4410" in _prompt(model)


def test_the_user_id_never_reaches_the_prompt(store, path, tracing):
    alice = f"{_uid('alice')}@example.com"
    store.add(Fact(scope="user", scope_id=alice, fact="prefers email"))
    model = TestModel(response="noted")
    agent = Agent(name="a", llm=model, memory=_per_user(store))
    _turn(agent, "hi", alice, path)
    assert "prefers email" in _prompt(model) and alice not in _prompt(model)


def test_learned_facts_are_capped_and_yours_are_kept(store, path, tracing):
    """Also: a message stores at most 10 facts, and learned facts say so."""
    alice = _uid("alice")

    def extractor(messages):
        text = (messages[-1].content or "").split("Message:", 1)[1].split("\n\nJSON:")[0]
        return json.dumps([f"{text.strip()} / fact {i}" for i in range(12)])

    mem = _per_user(store, learn=FunctionModel(extractor), max_learned_facts=5)
    mem.persist("typed in by hand", tier="user", id=alice)
    agent = Agent(name="a", llm=TestModel(response="noted"), memory=mem)
    for i in range(3):
        _turn(agent, f"message {i}", alice, path)
    facts = store.list_active(scope="user", scope_id=alice)
    learned = [f for f in facts if f.fact != "typed in by hand"]
    assert len(learned) == 5  # 3 messages x 10 kept, capped at 5
    assert all(f.source == "learned" for f in learned)
    assert any(f.fact == "typed in by hand" for f in facts)


def test_run_and_astream_store_the_same_turn(tracing):
    """The final reply after middleware, never text said before a tool call; the
    user's words as said."""

    def lookup() -> str:
        """Look the address up."""
        return "bob@example.com"

    def responder(messages):
        if any(m.role == MessageRole.tool for m in messages):
            return "Write to bob@example.com."
        return ("Let me check. ", [{"name": "lookup", "arguments": {}}])

    stored: dict[str, list[tuple[str, str]]] = {}
    for path in ("run", "astream"):
        store_mem = Memory(user_id=lambda ctx: ctx.state.user_id)
        agent = Agent(
            name="a",
            llm=FunctionModel(responder),
            tools=[FunctionTool(name="lookup", fn=lookup)],
            middleware=[RedactPII()],
            memory=store_mem,
        )
        alice = _uid("alice")
        _turn(agent, "My email is carol@example.com", alice, path)
        stored[path] = [(m.role.value, m.content or "") for m in store_mem.for_user(alice).messages]
    assert stored["run"] == stored["astream"]
    (user, said), (assistant, reply) = stored["run"]
    assert said == "My email is carol@example.com"
    assert "Let me check" not in reply and "bob@example.com" not in reply


@pytest.mark.parametrize("memory_kind", ["Memory", "ComposableMemory"])
def test_an_optimized_agent_keeps_users_apart_and_keeps_their_facts(store, tracing, memory_kind):
    from fastaiagent.agent.memory import AgentMemory, ComposableMemory
    from fastaiagent.agent.memory_blocks import PersistentFactBlock
    from fastaiagent.optimize import Candidate, apply_candidate

    alice, bob, agent_id = _uid("alice"), _uid("bob"), _uid("kb")
    store.add(Fact(scope="user", scope_id=alice, fact="alice-fact-4410"))
    shared = store.add(Fact(scope="agent", scope_id=agent_id, fact="cite sources"))
    if memory_kind == "Memory":
        mem: Any = _per_user(store, agent_id=agent_id)
    else:
        mem = ComposableMemory(
            blocks=[
                PersistentFactBlock(scope="agent", scope_id=agent_id, store=store),
                PersistentFactBlock(
                    scope="user", scope_id=lambda ctx: ctx.state.user_id, store=store
                ),
            ],
            primary=AgentMemory(),
        )
    model = TestModel(response="noted")
    base = Agent(name=agent_id, llm=model, memory=mem)
    optimized = apply_candidate(base, Candidate(fact_ids=[shared]), allow_writable_memory=True)

    _turn(optimized, "hi", alice, "run")
    seen = _prompt(model)
    assert "alice-fact-4410" in seen and "cite sources" in seen
    _turn(optimized, "hello", bob, "run")
    assert "alice-fact-4410" not in _prompt(model)
    if memory_kind == "Memory":
        # A per-user Memory keeps a window per user; a hand-built
        # ComposableMemory has one window for everyone, by design.
        assert "hi" not in [m.content for m in optimized.memory.for_user(bob).messages]
