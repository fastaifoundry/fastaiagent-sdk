"""Memory facade + safe-by-default scoping + dynamic scope_id.

No mocking: real SQLite ``MemoryStore``, real blocks, real tracing. No LLM
(the facade's `persist` is verbatim; extraction is covered by the e2e gate).
"""

from __future__ import annotations

import asyncio
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest

from fastaiagent import Agent
from fastaiagent._internal.config import reset_config
from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.agent.context import (
    RunContext,
    get_active_run_context,
    reset_active_run_context,
    set_active_run_context,
)
from fastaiagent.agent.memory_blocks import PersistentFactBlock
from fastaiagent.agent.memory_simple import Memory
from fastaiagent.guardrail import Guardrail, GuardrailPosition
from fastaiagent.learn import Fact, MemoryStore
from fastaiagent.llm.message import UserMessage
from fastaiagent.testing import TestModel
from fastaiagent.trace.otel import get_tracer
from fastaiagent.trace.otel import reset as reset_tracer


@pytest.fixture
def db(tmp_path: Path, monkeypatch):
    p = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(p))
    reset_config()
    reset_tracer()
    yield p
    reset_tracer()
    reset_config()


@dataclass
class St:
    user_id: str


# ---------------------------------------------------------------------------
# Safe-by-default scoping (the leak fix)
# ---------------------------------------------------------------------------


def test_safe_default_user_scope(db):
    s = MemoryStore(db_path=str(db))
    s.add(Fact(scope="user", scope_id="alice", fact="a-fact"))
    s.add(Fact(scope="user", scope_id="bob", fact="b-fact"))
    # empty id at user scope → nothing (was: everyone)
    assert s.list_active(scope="user", scope_id="") == []
    # explicit "*" → all
    assert {f.fact for f in s.list_active(scope="user", scope_id="*")} == {"a-fact", "b-fact"}
    # specific id → that subject only
    assert [f.fact for f in s.list_active(scope="user", scope_id="alice")] == ["a-fact"]


def test_agent_scope_stays_permissive(db):
    s = MemoryStore(db_path=str(db))
    s.add(Fact(scope="agent", scope_id="x", fact="global-1"))
    s.add(Fact(scope="agent", scope_id="y", fact="global-2"))
    # agent is the global tier: empty id = all (unchanged)
    assert {f.fact for f in s.list_active(scope="agent", scope_id="")} == {"global-1", "global-2"}


def test_delete_refuses_mass_delete_without_id(db):
    s = MemoryStore(db_path=str(db))
    s.add(Fact(scope="user", scope_id="alice", fact="keep"))
    with pytest.raises(ValueError, match="explicit scope_id"):
        s.delete(scope="user", scope_id="")
    assert s.list_active(scope="user", scope_id="alice")  # untouched


# ---------------------------------------------------------------------------
# Dynamic scope_id + cross-session isolation (the multi-user guard)
# ---------------------------------------------------------------------------


def test_dynamic_scope_id_isolates_users(db):
    s = MemoryStore(db_path=str(db))
    s.add(Fact(scope="user", scope_id="alice", fact="Alice likes email"))
    s.add(Fact(scope="user", scope_id="bob", fact="Bob likes SMS"))
    block = PersistentFactBlock(scope="user", scope_id=lambda ctx: ctx.state.user_id, store=s)

    def render_for(uid: str) -> str:
        tok = set_active_run_context(RunContext(state=St(user_id=uid)))
        try:
            out = block.render("?")
            return out[0].content if out else ""
        finally:
            reset_active_run_context(tok)

    assert "Alice likes email" in render_for("alice")
    assert "Bob likes SMS" in render_for("bob")
    # switching back must NOT serve bob's cached facts (cache invalidation)
    assert "Alice likes email" in render_for("alice")
    assert "Bob" not in render_for("alice")


def test_dynamic_scope_id_no_context_is_safe(db):
    s = MemoryStore(db_path=str(db))
    s.add(Fact(scope="user", scope_id="alice", fact="secret"))
    block = PersistentFactBlock(scope="user", scope_id=lambda ctx: ctx.state.user_id, store=s)
    # No active RunContext → resolver yields "" → no personal facts.
    assert block.render("?") == []


# ---------------------------------------------------------------------------
# Memory facade — direct verbs
# ---------------------------------------------------------------------------


def test_persist_retrieve_forget_roundtrip(db):
    mem = Memory(location=MemoryStore(db_path=str(db)), agent_id="support")
    mem.persist("Return policy is 30 days", tier="global")
    fid = mem.persist("Alice prefers email", tier="user", id="alice")
    assert isinstance(fid, int)
    assert [f.fact for f in mem.retrieve(tier="global")] == ["Return policy is 30 days"]
    assert [f.fact for f in mem.retrieve(tier="user", id="alice")] == ["Alice prefers email"]
    # safe: user tier without id → []
    assert mem.retrieve(tier="user") == []
    # forget
    assert mem.forget(tier="user", id="alice") == 1
    assert mem.retrieve(tier="user", id="alice") == []


def test_persist_returns_fact_objects_with_metadata(db):
    mem = Memory(location=MemoryStore(db_path=str(db)))
    mem.persist("fact one", tier="user", id="u", confidence=0.6)
    rows = mem.retrieve(tier="user", id="u")
    assert isinstance(rows[0], Fact)
    assert rows[0].confidence == 0.6
    assert rows[0].source_trace_id is None  # direct persist = "manual"


def test_tier_validation(db):
    mem = Memory(location=MemoryStore(db_path=str(db)))
    with pytest.raises(ValueError, match="requires id"):
        mem.persist("x", tier="user")
    with pytest.raises(NotImplementedError):
        mem.retrieve(tier="session")
    with pytest.raises(NotImplementedError):
        mem.retrieve("semantic?", tier="user", id="u")
    with pytest.raises(ValueError, match="tier must be"):
        mem.retrieve(tier="bogus")


# ---------------------------------------------------------------------------
# Memory facade — agent-attachable contract
# ---------------------------------------------------------------------------


def test_memory_is_drop_in_window(db):
    mem = Memory(location=MemoryStore(db_path=str(db)), window=10)
    mem.add(UserMessage("hello"))
    ctx = mem.get_context("?")
    assert [m.content for m in ctx] == ["hello"]
    assert len(mem) == 1
    assert bool(mem) is True


def test_bare_memory_has_no_fact_blocks(db):
    mem = Memory(location=MemoryStore(db_path=str(db)))  # no user_id/agent_id
    assert mem.blocks == []  # window only, no surprise DB reads


def test_dynamic_memory_routes_per_user_no_window_leak(db):
    """One Memory, dynamic user_id: durable facts AND the live window are
    isolated per user (the multi-session guarantee)."""
    store = MemoryStore(db_path=str(db))
    store.add(Fact(scope="user", scope_id="alice", fact="Alice fact"))
    store.add(Fact(scope="user", scope_id="bob", fact="Bob fact"))
    mem = Memory(location=store, user_id=lambda ctx: ctx.state.user_id)

    def run_turn(uid: str, text: str) -> list[str]:
        tok = set_active_run_context(RunContext(state=St(user_id=uid)))
        try:
            mem.add(UserMessage(text))  # goes into THIS user's window
            return [m.content or "" for m in mem.get_context("?")]
        finally:
            reset_active_run_context(tok)

    alice_ctx = run_turn("alice", "alice-private-message")
    bob_ctx = run_turn("bob", "bob-private-message")

    joined_alice = " ".join(alice_ctx)
    joined_bob = " ".join(bob_ctx)
    # durable facts isolated
    assert "Alice fact" in joined_alice and "Bob fact" not in joined_alice
    assert "Bob fact" in joined_bob and "Alice fact" not in joined_bob
    # live window isolated — bob never sees alice's message and vice versa
    assert "alice-private-message" not in joined_bob
    assert "bob-private-message" not in joined_alice
    # switching back to alice: her window persists, no bob bleed
    alice_again = run_turn("alice", "second-alice-message")
    j = " ".join(alice_again)
    assert "alice-private-message" in j and "bob-private-message" not in j


# ---------------------------------------------------------------------------
# Direct-op spans
# ---------------------------------------------------------------------------


def _read_spans(db_path: Path):
    reset_tracer()
    with SQLiteHelper(db_path) as d:
        return d.fetchall("SELECT name, attributes FROM spans ORDER BY start_time")


def test_persist_and_retrieve_emit_spans(db):
    mem = Memory(location=MemoryStore(db_path=str(db)), agent_id="support")
    tracer = get_tracer("fastaiagent")
    with tracer.start_as_current_span("agent.test"):
        mem.persist("g", tier="global")
        mem.retrieve(tier="global")

    spans = {r["name"]: json.loads(r["attributes"]) for r in _read_spans(db)}
    assert "memory.persist" in spans
    assert "memory.retrieve" in spans
    assert spans["memory.persist"]["memory.tier"] == "global"
    assert spans["memory.persist"]["memory.scope"] == "agent"
    assert spans["memory.persist"]["memory.count"] == 1
    assert spans["memory.retrieve"]["memory.count"] == 1


# ---------------------------------------------------------------------------
# 1.80.0 — one user's memory never reaches another
# ---------------------------------------------------------------------------


def _per_user(db: Path, **kwargs) -> Memory:
    return Memory(
        location=MemoryStore(db_path=str(db)),
        user_id=lambda ctx: ctx.state.user_id,
        **kwargs,
    )


def _turn(mem: Memory, context: RunContext | None, text: str) -> str:
    """One simulated turn: read (as the agent does), then write the user text."""
    tok = set_active_run_context(context)
    try:
        seen = " ".join(m.content or "" for m in mem.get_context("?"))
        mem.add(UserMessage(text))
        return seen
    finally:
        reset_active_run_context(tok)


def test_unresolved_callers_share_no_window(db):
    """Two callers with no RunContext used to share one anonymous window."""
    mem = _per_user(db)
    _turn(mem, None, "zed-private-message")
    seen = _turn(mem, None, "second caller")
    assert "zed-private-message" not in seen
    assert len(mem) == 0 and mem.messages == []


def test_dict_state_resolver_is_unresolved_not_shared(db, caplog):
    """``ctx.state.user_id`` on a dict state raises; those callers must not
    pool into one window, and the misconfiguration is logged once."""
    mem = _per_user(db)
    with caplog.at_level(logging.WARNING, logger="fastaiagent.agent.memory_blocks"):
        _turn(mem, RunContext(state={"user_id": "carol"}), "quinn-is-carols-secret")
        seen = _turn(mem, RunContext(state={"user_id": "dave"}), "dave here")
        _turn(mem, RunContext(state={"user_id": "erin"}), "erin here")
    assert "quinn-is-carols-secret" not in seen
    warnings = [r for r in caplog.records if "user_id resolver raised" in r.getMessage()]
    assert len(warnings) == 1


def test_resolver_returning_none_is_unresolved(db):
    """A resolver that returns None used to become the literal user "None",
    one bucket shared by every such caller."""
    mem = Memory(location=MemoryStore(db_path=str(db)), user_id=lambda ctx: None)
    _turn(mem, RunContext(state=St(user_id="x")), "first-none-caller-secret")
    seen = _turn(mem, RunContext(state=St(user_id="y")), "second none caller")
    assert "first-none-caller-secret" not in seen


def test_scope_resolver_returning_none_reads_nothing(db):
    s = MemoryStore(db_path=str(db))
    s.add(Fact(scope="user", scope_id="None", fact="fact filed under the string None"))
    block = PersistentFactBlock(scope="user", scope_id=lambda ctx: None, store=s)
    tok = set_active_run_context(RunContext(state=St(user_id="x")))
    try:
        assert block.render("?") == []
    finally:
        reset_active_run_context(tok)


def test_unresolved_caller_still_sees_global_facts(db):
    mem = _per_user(db, agent_id="support")
    mem.persist("Return policy is 30 days", tier="global")
    seen = _turn(mem, None, "hello")
    assert "Return policy is 30 days" in seen


def test_unresolved_agent_scope_resolver_reads_nothing(db):
    """An agent-scope block whose resolver can't resolve used to read
    ``scope_id=""`` — which at agent scope means every agent's facts."""
    s = MemoryStore(db_path=str(db))
    s.add(Fact(scope="agent", scope_id="billing", fact="billing-only fact"))
    s.add(Fact(scope="agent", scope_id="support", fact="support-only fact"))
    block = PersistentFactBlock(scope="agent", scope_id=lambda ctx: ctx.state.agent, store=s)
    assert block.render("?") == []  # no context
    tok = set_active_run_context(RunContext(state={"agent": "support"}))  # resolver raises
    try:
        assert block.render("?") == []
    finally:
        reset_active_run_context(tok)


def test_static_empty_agent_scope_warns_and_star_is_explicit(db):
    """A static ``scope_id=""`` at agent scope still reads every agent (the
    documented global read), but it is almost always a missing id, so it warns.
    ``"*"`` says the same thing on purpose, and is silent."""
    s = MemoryStore(db_path=str(db))
    s.add(Fact(scope="agent", scope_id="billing", fact="billing-only fact"))
    with pytest.warns(UserWarning, match="every agent"):
        block = PersistentFactBlock(scope="agent", scope_id="", store=s)
    out = block.render("?")
    assert out and "billing-only fact" in out[0].content

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        star = PersistentFactBlock(scope="agent", scope_id="*", store=s)
    out = star.render("?")
    assert out and "billing-only fact" in out[0].content


def test_empty_agent_id_is_refused(db):
    """``Memory(agent_id="")`` read every agent's global facts — usually an
    unset setting, e.g. ``os.environ.get("AGENT_ID", "")``."""
    with pytest.raises(ValueError, match="every agent"):
        Memory(location=MemoryStore(db_path=str(db)), agent_id="")


def test_save_load_outside_a_run_point_to_for_user(db, tmp_path):
    mem = _per_user(db)
    _turn(mem, RunContext(state=St(user_id="alice")), "alice-window-message")
    with pytest.raises(ValueError, match="for_user"):
        mem.save(tmp_path / "m")
    with pytest.raises(ValueError, match="for_user"):
        mem.load(tmp_path / "m")
    mem.clear()  # no current user: nothing to clear, and no error

    mem.for_user("alice").save(tmp_path / "alice")
    assert "alice-window-message" in (tmp_path / "alice" / "primary.json").read_text()

    fresh = _per_user(db)
    fresh.for_user("alice").load(tmp_path / "alice")
    seen = _turn(fresh, RunContext(state=St(user_id="alice")), "back again")
    assert "alice-window-message" in seen


def test_for_user_needs_a_per_user_memory_and_an_id(db):
    with pytest.raises(ValueError):
        _per_user(db).for_user("")
    with pytest.raises(ValueError):
        Memory(location=MemoryStore(db_path=str(db))).for_user("alice")


def test_forget_global_without_id_refuses_to_delete_every_agent(db):
    s = MemoryStore(db_path=str(db))
    s.add(Fact(scope="agent", scope_id="billing", fact="billing-only fact"))
    s.add(Fact(scope="agent", scope_id="support", fact="support-only fact"))
    mem = Memory(location=s)  # no agent_id
    with pytest.raises(ValueError, match='id="\\*"'):
        mem.forget(tier="global")
    with pytest.raises(ValueError, match='id="\\*"'):
        mem.forget(tier="global", fact="billing-only fact")
    assert len(s.list_active(scope="agent", scope_id="")) == 2  # untouched

    # Scoped by agent_id: only that agent's facts go.
    assert Memory(location=s, agent_id="support").forget(tier="global") == 1
    # Explicit opt-in still works.
    assert mem.forget(tier="global", id="*") == 1


# --- streaming (Agent.astream) ----------------------------------------------


def _prompt(call: dict) -> str:
    return " ".join(m.content or "" for m in call["messages"] if isinstance(m.content, str))


async def _drain(agent: Agent, text: str, uid: str) -> None:
    async for _ in agent.astream(text, context=RunContext(state=St(user_id=uid))):
        pass


def test_astream_reads_the_callers_own_window(db):
    """``astream`` built the prompt before exposing the RunContext, so a
    per-user Memory read the anonymous window while writing to the user's."""
    model = TestModel(response="noted")
    agent = Agent(name="support", llm=model, memory=_per_user(db))

    async def go():
        await _drain(agent, "alice-first-message", "alice")
        await _drain(agent, "alice-second-message", "alice")
        await _drain(agent, "bob-first-message", "bob")

    asyncio.run(go())
    assert "alice-first-message" in _prompt(model.calls[1])
    assert "alice-first-message" not in _prompt(model.calls[2])
    assert "alice-second-message" not in _prompt(model.calls[2])


def test_interleaved_streams_write_to_their_own_users(db):
    """Two ``astream`` generators advanced in one task share that task's
    context; each must still write its turn to its own user."""
    mem = _per_user(db)
    agent = Agent(name="support", llm=TestModel(response="noted"), memory=mem)

    async def go():
        ga = agent.astream("alice-turn", context=RunContext(state=St(user_id="alice")))
        gb = agent.astream("bob-turn", context=RunContext(state=St(user_id="bob")))
        await ga.__anext__()
        await gb.__anext__()
        async for _ in ga:
            pass
        async for _ in gb:
            pass

    asyncio.run(go())
    alice = " ".join(m.content or "" for m in mem.for_user("alice").messages)
    bob = " ".join(m.content or "" for m in mem.for_user("bob").messages)
    assert "alice-turn" in alice and "bob-turn" not in alice
    assert "bob-turn" in bob and "alice-turn" not in bob


def test_blocked_stream_leaves_no_run_context(db):
    guard = Guardrail(
        name="block_bad",
        position=GuardrailPosition.input,
        blocking=True,
        fn=lambda text: "bad" not in text,
    )
    agent = Agent(name="a", llm=TestModel(), guardrails=[guard], memory=_per_user(db))

    async def go():
        with pytest.raises(GuardrailBlockedError):
            await _drain(agent, "bad input", "alice")
        return get_active_run_context()

    assert asyncio.run(go()) is None


def test_store_span_keeps_scope_id_locally_with_payloads_off(db, monkeypatch):
    """The egress gate never touches local capture (CLAUDE.md §2.5)."""
    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")
    mem = Memory(location=MemoryStore(db_path=str(db)))
    with get_tracer("fastaiagent").start_as_current_span("agent.test"):
        mem.persist("x", tier="user", id="alice@example.com")
    spans = {r["name"]: json.loads(r["attributes"]) for r in _read_spans(db)}
    assert spans["memory.persist"]["memory.scope_id"] == "alice@example.com"


# --- 1.81.0: distinct block names, window pairs, no eager .fastaiagent ------------


def test_agent_and_user_fact_blocks_have_distinct_names(db):
    """Both were named "persistent_facts": one span label, one by_block key, and
    optimize's block replacement removed both."""
    mem = _per_user(db, agent_id="support")
    names = [b.name for b in mem.for_user("alice").blocks]
    assert "persistent_facts" in names and "persistent_facts.user" in names

    store = MemoryStore(db_path=str(db))
    store.add(Fact(scope="agent", scope_id="support", fact="global fact"))
    store.add(Fact(scope="user", scope_id="alice", fact="alice fact"))
    agent = Agent(name="support", llm=TestModel(response="ok"), memory=mem)
    agent.run("hi", context=RunContext(state=St(user_id="alice")))
    span_names = {r["name"] for r in _read_spans(db)}
    assert {"memory.read.persistent_facts", "memory.read.persistent_facts.user"} <= span_names


def test_an_odd_window_never_starts_with_an_orphan_reply():
    from fastaiagent import AgentMemory
    from fastaiagent.llm.message import AssistantMessage

    mem = AgentMemory(max_messages=3)
    for i in (1, 2):
        mem.add(UserMessage(f"u{i}"))
        mem.add(AssistantMessage(f"a{i}"))
    assert [m.content for m in mem.messages] == ["u2", "a2"]  # was ["a1", "u2", "a2"]


def test_memory_creates_no_local_files_until_used(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FASTAIAGENT_LOCAL_DB", raising=False)
    reset_config()
    try:
        mem = Memory()
        assert not (tmp_path / ".fastaiagent").exists()
        mem.persist("x", tier="user", id="u")
        assert (tmp_path / ".fastaiagent" / "local.db").exists()
    finally:
        reset_config()


# --- 1.82.0: per-user windows are bounded --------------------------------------


def test_per_user_windows_evict_the_least_recently_used(db, caplog):
    """``_per_user`` kept every user's window for the life of the process."""
    mem = _per_user(db, max_users=2)
    _turn(mem, RunContext(state=St(user_id="alice")), "alice-message")
    _turn(mem, RunContext(state=St(user_id="bob")), "bob-message")
    # Touching alice makes bob the least recently used.
    _turn(mem, RunContext(state=St(user_id="alice")), "alice again")
    with caplog.at_level(logging.WARNING, logger="fastaiagent.agent.memory_simple"):
        _turn(mem, RunContext(state=St(user_id="carol")), "carol-message")
        _turn(mem, RunContext(state=St(user_id="dave")), "dave-message")
    assert set(mem._per_user) == {"carol", "dave"}
    evictions = [r for r in caplog.records if "max_users" in r.getMessage()]
    assert len(evictions) == 1  # warned once, not per eviction


def test_touching_a_user_keeps_their_window(db):
    mem = _per_user(db, max_users=2)
    _turn(mem, RunContext(state=St(user_id="alice")), "alice-message")
    _turn(mem, RunContext(state=St(user_id="bob")), "bob-message")
    _turn(mem, RunContext(state=St(user_id="alice")), "alice again")
    _turn(mem, RunContext(state=St(user_id="carol")), "carol-message")
    seen = _turn(mem, RunContext(state=St(user_id="alice")), "third")
    assert "alice-message" in seen


def test_an_evicted_user_keeps_their_facts(db):
    mem = _per_user(db, max_users=1)
    mem.persist("alice likes tea", tier="user", id="alice")
    _turn(mem, RunContext(state=St(user_id="alice")), "alice-message")
    _turn(mem, RunContext(state=St(user_id="bob")), "bob-message")
    seen = _turn(mem, RunContext(state=St(user_id="alice")), "back again")
    assert "alice-message" not in seen  # the window went
    assert "alice likes tea" in seen  # the durable fact did not


def test_max_users_must_be_positive(db):
    with pytest.raises(ValueError, match="max_users"):
        _per_user(db, max_users=0)
    assert len(_per_user(db, max_users=None)._per_user) == 0  # None = unbounded


def test_first_calls_for_one_user_share_one_window(db):
    """An unlocked get-or-create let two threads build two windows for one user,
    and the loser's messages vanished."""
    import threading
    import time

    pytest.importorskip("faiss")

    class _SlowEmbedder:  # a real, slow embedder widens the race window
        def embed(self, texts):
            time.sleep(0.05)
            return [[float(len(t)), 1.0, 0.0] for t in texts]

    mem = _per_user(db, recall="auto", embedder=_SlowEmbedder())
    got: list = []
    threads = [threading.Thread(target=lambda: got.append(mem.for_user("alice"))) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({id(m) for m in got}) == 1


# --- 1.82.0: the user id stays out of the prompt --------------------------------


def test_the_user_id_is_not_written_into_the_prompt(db):
    """The user fact block's heading was "Learned facts (user:<id>)", so an
    email id went to the model provider on every turn — even with trace payloads
    off, which gates export, not prompts."""
    store = MemoryStore(db_path=str(db))
    store.add(Fact(scope="user", scope_id="alice@example.com", fact="Prefers email"))
    store.add(Fact(scope="agent", scope_id="support", fact="Returns take 30 days"))
    mem = _per_user(db, agent_id="support")
    model = TestModel(response="ok")
    agent = Agent(name="support", llm=model, memory=mem)
    agent.run("hi", context=RunContext(state=St(user_id="alice@example.com")))
    sent = _prompt(model.calls[-1])
    assert "Prefers email" in sent and "Learned facts (user):" in sent
    assert "alice@example.com" not in sent
    assert "Learned facts (agent:support):" in sent  # an agent id is not personal
