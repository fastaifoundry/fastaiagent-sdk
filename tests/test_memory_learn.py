"""``learn=`` stores clean user facts, once (1.81.0).

No mocking: a real SQLite ``MemoryStore``, real blocks and the real ``Memory``
facade. The extractor is the SDK's own offline ``FunctionModel`` so extraction
calls can be counted; the agent is a ``TestModel``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastaiagent import Agent, Memory, RunContext
from fastaiagent._internal.config import reset_config
from fastaiagent.agent.memory_blocks import FactExtractionBlock
from fastaiagent.learn import MemoryStore
from fastaiagent.llm.message import AssistantMessage, UserMessage
from fastaiagent.testing import FunctionModel, TestModel
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


def _extractor() -> FunctionModel:
    """One fact per message: a user message yields a user fact, anything else
    yields a claim the assistant made."""

    def responder(messages):
        prompt = messages[-1].content or ""
        text = prompt.split("Message:", 1)[1].split("\n\nJSON:", 1)[0].strip()
        if text.startswith("I "):
            return json.dumps([f"User says: {text}"])
        return json.dumps([f"Assistant claimed: {text}"])

    return FunctionModel(responder)


def _ctx(uid: str) -> RunContext:
    return RunContext(state={"uid": uid})


def _learned(db: Path, uid: str) -> list[str]:
    return [f.fact for f in MemoryStore(db_path=str(db)).list_active(scope="user", scope_id=uid)]


# --- B12: learn= reads the user's messages, not the model's replies ----------


def test_learn_extracts_from_user_messages_only(db):
    extractor = _extractor()
    mem = Memory(
        location=MemoryStore(db_path=str(db)),
        user_id=lambda ctx: ctx.state["uid"],
        learn=extractor,
    )
    agent = Agent(name="a", llm=TestModel(response="Biscuit is allergic to cats."), memory=mem)
    agent.run("I have a beagle named Biscuit", context=_ctx("u1"))

    assert len(extractor.calls) == 1  # was 2: one per user and assistant message
    facts = _learned(db, "u1")
    assert facts == ["User says: I have a beagle named Biscuit"]
    assert not any("Assistant claimed" in f for f in facts)


def test_fact_extraction_block_still_reads_both_roles_by_default():
    extractor = _extractor()
    block = FactExtractionBlock(llm=extractor)
    block.on_message(UserMessage("I live in Lisbon"))
    block.on_message(AssistantMessage("The MAAT museum is in Lisbon"))
    assert len(extractor.calls) == 2
    assert block._facts == [
        "User says: I live in Lisbon",
        "Assistant claimed: The MAAT museum is in Lisbon",
    ]


def test_fact_extraction_block_roles_are_validated():
    with pytest.raises(ValueError, match="roles"):
        FactExtractionBlock(llm=_extractor(), roles=())
    with pytest.raises(ValueError, match="roles"):
        FactExtractionBlock(llm=_extractor(), roles=("tool",))


# --- B13: each learned fact reaches the prompt once ----------------------------


def test_learned_facts_are_injected_once(db):
    model = TestModel(response="noted")
    mem = Memory(
        location=MemoryStore(db_path=str(db)),
        user_id=lambda ctx: ctx.state["uid"],
        learn=_extractor(),
    )
    agent = Agent(name="a", llm=model, memory=mem)
    agent.run("I have a beagle named Biscuit", context=_ctx("u1"))
    agent.run("What is my dog's name?", context=_ctx("u1"))

    system_text = " ".join(
        m.content or "" for m in model.calls[1]["messages"] if m.role.value == "system"
    )
    # was 2: "Known facts" (FactExtractionBlock) + "Learned facts" (the store)
    assert system_text.count("User says: I have a beagle named Biscuit") == 1


def test_fact_extraction_block_with_inject_false_renders_nothing():
    block = FactExtractionBlock(llm=_extractor(), inject=False)
    block.on_message(UserMessage("I live in Lisbon"))
    assert block._facts  # still extracted
    assert block.render("where do I live?") == []
    report = block.last_render_report()
    assert report is not None and report.rendered_count == 0


# --- Cap: a user's auto-learned facts stay bounded -----------------------------


def test_learned_facts_are_capped_and_manual_facts_are_kept(db):
    store = MemoryStore(db_path=str(db))
    mem = Memory(
        location=store,
        user_id=lambda ctx: ctx.state["uid"],
        learn=_extractor(),
        max_learned_facts=3,
    )
    mem.persist("Account manager is Dana", tier="user", id="u1")  # yours: never pruned
    agent = Agent(name="a", llm=TestModel(response="ok"), memory=mem)
    for i in range(5):
        agent.run(f"I note item {i}", context=_ctx("u1"))

    facts = _learned(db, "u1")
    learned = [f for f in facts if f.startswith("User says:")]
    assert learned == [f"User says: I note item {i}" for i in (4, 3, 2)]  # newest three
    assert "Account manager is Dana" in facts


def test_block_reports_what_the_cap_pruned(db):
    store = MemoryStore(db_path=str(db))
    block = FactExtractionBlock(
        llm=_extractor(),
        persist=True,
        scope="user",
        scope_id="u1",
        store=store,
        max_persisted=2,
    )
    with get_tracer("fastaiagent").start_as_current_span("agent.test"):
        for i in range(3):
            block.on_message(UserMessage(f"I note item {i}"))
    report = block.last_write_report()
    assert report is not None and report.detail["pruned"] == 1
    assert len(store.list_active(scope="user", scope_id="u1")) == 2


def test_max_persisted_is_validated():
    with pytest.raises(ValueError, match="max_persisted"):
        FactExtractionBlock(llm=_extractor(), max_persisted=0)


# --- B14: a global fact without an agent id is never injected ------------------


def test_persisting_a_global_fact_without_agent_id_warns(db):
    mem = Memory(location=MemoryStore(db_path=str(db)))
    with pytest.warns(UserWarning, match="agent_id"):
        mem.persist("Support replies within 24 hours.", tier="global")
    with pytest.warns(UserWarning, match="agent_id"):
        mem.update(
            "Support replies within 12 hours.",
            old="Support replies within 24 hours.",
            tier="global",
        )


def test_persisting_a_global_fact_with_agent_id_is_injected(db):
    import warnings

    store = MemoryStore(db_path=str(db))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Memory(location=store, agent_id="support").persist(
            "Support replies within 24 hours.", tier="global"
        )
    seen = " ".join(
        m.content or "" for m in Memory(location=store, agent_id="support").get_context("?")
    )
    assert "Support replies within 24 hours." in seen
