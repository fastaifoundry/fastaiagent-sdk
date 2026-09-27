"""Fast, deterministic unit tests for ``fastaiagent.optimize`` (no LLM, no mocks).

The LLM-dependent loop is covered end-to-end in
``tests/e2e/test_optimize_e2e.py``. Here we test the pure scaffolding: the seeded
split, the clone-and-patch seam, score roll-up math, the Contract-3/4 guards,
config validation, and report rendering — all with real objects.
"""

from __future__ import annotations

import asyncio
import warnings
from dataclasses import dataclass

import pytest

from fastaiagent import Agent, LLMClient
from fastaiagent.agent.context import RunContext
from fastaiagent.eval.llm_judge import LLMJudge
from fastaiagent.eval.results import EvalResults
from fastaiagent.eval.scorer import ScorerResult
from fastaiagent.optimize import (
    Candidate,
    CandidateScore,
    OptimizationReport,
    OptimizeConfig,
    apply_candidate,
)
from fastaiagent.optimize.candidate import _clone_memory_blocks, scorer_present
from fastaiagent.optimize.loop import _split
from fastaiagent.optimize.proposers import propose_prompt_rewrites
from fastaiagent.optimize.report import TrajectoryPoint
from fastaiagent.testing import TestModel


def _agent(prompt: str = "ORIGINAL") -> Agent:
    # Construction is offline — no API key needed (key is only used at call time).
    return Agent(
        name="t",
        system_prompt=prompt,
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    )


# ── Candidate + clone-and-patch seam ────────────────────────────────────────


def test_candidate_holds_all_three_levers():
    c = Candidate(system_prompt="p")
    # Frozen data model: all three lever fields exist; only system_prompt set in P1.
    assert c.system_prompt == "p"
    assert c.fewshot_demos is None
    assert c.fact_ids is None
    assert c.id and isinstance(c.id, str)
    assert set(c.to_dict()) == {
        "id",
        "parent_id",
        "origin",
        "rationale",
        "system_prompt",
        "fewshot_demos",
        "fact_ids",
    }


def test_apply_candidate_patches_prompt_without_mutating_base():
    base = _agent("ORIGINAL")
    patched = apply_candidate(base, Candidate(system_prompt="NEW"))
    assert patched.system_prompt == "NEW"
    assert base.system_prompt == "ORIGINAL"  # base untouched
    assert patched is not base
    # carries through llm / tools / config identity
    assert patched.llm is base.llm
    assert patched.config is base.config


def test_apply_candidate_inherits_prompt_when_none():
    base = _agent("ORIGINAL")
    patched = apply_candidate(base, Candidate())  # system_prompt None → inherit
    assert patched.system_prompt == "ORIGINAL"


# ── CandidateScore roll-up ──────────────────────────────────────────────────


def _eval_results(pairs: dict[str, list[tuple[float, bool]]]) -> EvalResults:
    return EvalResults(
        scores={
            name: [ScorerResult(score=s, passed=p) for s, p in vals] for name, vals in pairs.items()
        }
    )


def test_candidate_score_defaults_to_overall_pass_rate():
    er = _eval_results({"exact_match": [(1.0, True), (0.0, False)]})
    cs = CandidateScore.from_eval("cid", "dev", er, primary_metric=None)
    assert cs.score == pytest.approx(0.5)  # 1 of 2 passed
    assert cs.per_metric == {"exact_match": pytest.approx(0.5)}
    assert cs.n == 2


def test_candidate_score_uses_primary_metric_avg():
    er = _eval_results(
        {"exact_match": [(1.0, True), (0.0, False)], "judge": [(0.8, True), (0.6, True)]}
    )
    cs = CandidateScore.from_eval("cid", "dev", er, primary_metric="judge")
    assert cs.score == pytest.approx(0.7)  # avg of judge scores


def test_candidate_score_missing_primary_metric_warns_and_falls_back():
    er = _eval_results({"exact_match": [(1.0, True)]})
    with pytest.warns(UserWarning, match="not among scored metrics"):
        cs = CandidateScore.from_eval("cid", "dev", er, primary_metric="nope")
    assert cs.score == pytest.approx(1.0)  # overall pass-rate fallback


# ── Seeded split ────────────────────────────────────────────────────────────


def test_split_is_deterministic_and_partitions():
    items = [{"input": str(i)} for i in range(20)]
    a = _split(items, (0.5, 0.25, 0.25), seed=7)
    b = _split(items, (0.5, 0.25, 0.25), seed=7)
    assert a == b  # same seed → same split
    train, dev, hold = a
    assert len(train) + len(dev) + len(hold) == 20
    # disjoint coverage
    seen = [it["input"] for part in a for it in part]
    assert sorted(seen) == sorted(it["input"] for it in items)


def test_split_different_seed_differs():
    items = [{"input": str(i)} for i in range(20)]
    assert _split(items, (0.5, 0.25, 0.25), 1) != _split(items, (0.5, 0.25, 0.25), 2)


def test_split_guarantees_nonempty_each_for_small_n():
    items = [{"input": str(i)} for i in range(3)]
    train, dev, hold = _split(items, (0.5, 0.25, 0.25), seed=0)
    assert len(train) >= 1 and len(dev) >= 1 and len(hold) >= 1


# ── Contract 3: judge dedup ─────────────────────────────────────────────────


def test_scorer_present_identity_and_name():
    j = LLMJudge(criteria="x", name="myjudge")
    assert scorer_present([j], j) is True  # identity
    assert scorer_present([LLMJudge(criteria="y", name="myjudge")], j) is True  # name match
    assert scorer_present([LLMJudge(criteria="z", name="other")], j) is False
    assert scorer_present([], j) is False


# ── P2: memory isolation via isolated_copy() ────────────────────────────────


def test_isolated_copy_static_and_fewshot_are_fresh():
    from fastaiagent.agent.memory_blocks import FewShotBlock, StaticBlock

    s = StaticBlock("hi", name="s")
    s2 = s.isolated_copy()
    assert s2 is not s and s2.text == "hi" and s2.name == "s"

    f = FewShotBlock([{"input": "q", "output": "a"}], name="fs")
    f2 = f.isolated_copy()
    assert f2 is not f and f2.demos == f.demos and f2.name == "fs"


def test_isolated_copy_shares_handle_resets_state():
    from fastaiagent.agent.memory_blocks import FactExtractionBlock, SummaryBlock

    llm = _agent().llm
    s = SummaryBlock(llm=llm, keep_last=3)
    s2 = s.isolated_copy()
    assert s2.llm is llm and s2.keep_last == 3  # shared handle + config
    assert s2._archive == [] and s2._summary == "" and s2._messages_seen == 0  # fresh state

    fe = FactExtractionBlock(llm=llm, max_facts=7)
    fe2 = fe.isolated_copy()
    assert fe2.llm is llm and fe2.max_facts == 7 and fe2._facts == []


def test_isolated_copy_two_candidate_no_bleed():
    # Spec hard requirement: candidate A's in-process writes must not appear in B.
    from fastaiagent.agent.memory_blocks import SummaryBlock
    from fastaiagent.llm.message import UserMessage

    src = SummaryBlock(llm=_agent().llm)
    a, b = src.isolated_copy(), src.isolated_copy()
    a.on_message(UserMessage("candidate A turn"))
    assert len(a._archive) == 1 and len(b._archive) == 0  # no bleed


def test_isolated_copy_vectorblock_raises():
    from fastaiagent.agent.memory_blocks import MemoryIsolationError, VectorBlock

    class _Store:
        def add(self, *a, **k): ...
        def search(self, *a, **k):
            return []

    with pytest.raises(MemoryIsolationError):
        VectorBlock(store=_Store()).isolated_copy()


def test_default_isolated_copy_warns_and_shares():
    from fastaiagent.agent.memory_blocks import MemoryBlock

    class CustomBlock(MemoryBlock):
        name = "custom"

        def on_message(self, message): ...

        def render(self, query):
            return []

    b = CustomBlock()
    with pytest.warns(UserWarning, match="no isolated_copy"):
        assert b.isolated_copy() is b  # default: warn + share


def test_clone_memory_blocks_none_and_agentmemory():
    from fastaiagent.agent.memory import AgentMemory

    assert _clone_memory_blocks(None) is None
    m = AgentMemory(max_messages=5)
    c = _clone_memory_blocks(m)
    assert c is not m and c.max_messages == 5 and len(c) == 0


def test_clone_memory_blocks_composable_fresh_blocks_and_primary():
    from fastaiagent.agent.memory import AgentMemory, ComposableMemory
    from fastaiagent.agent.memory_blocks import StaticBlock

    src = ComposableMemory(blocks=[StaticBlock("x")], primary=AgentMemory(max_messages=9))
    c = _clone_memory_blocks(src)
    assert c is not src and c.blocks[0] is not src.blocks[0]
    assert c.blocks[0].text == "x" and c.primary.max_messages == 9


def test_clone_memory_blocks_vectorblock_refuse_then_allow():
    from fastaiagent.agent.memory import AgentMemory, ComposableMemory
    from fastaiagent.agent.memory_blocks import MemoryIsolationError, VectorBlock

    class _Store:
        def add(self, *a, **k): ...
        def search(self, *a, **k):
            return []

    src = ComposableMemory(blocks=[VectorBlock(store=_Store())], primary=AgentMemory())
    with pytest.raises(MemoryIsolationError):
        _clone_memory_blocks(src)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = _clone_memory_blocks(src, allow_writable_memory=True)
    assert c.blocks[0] is src.blocks[0]  # shared, not cloned


# ── P2: FewShotBlock render + few-shot injection ────────────────────────────


def test_fewshot_block_renders_demos():
    from fastaiagent.agent.memory_blocks import FewShotBlock

    out = FewShotBlock([{"input": "Capital of France?", "output": "Paris"}]).render("")
    assert len(out) == 1
    body = out[0].content
    assert "Capital of France?" in body and "Paris" in body
    assert FewShotBlock([]).render("") == []  # empty → nothing


def test_apply_candidate_injects_fewshot_without_stacking():
    from fastaiagent.agent.memory import ComposableMemory

    base = _agent("P")
    c1 = apply_candidate(base, Candidate(fewshot_demos=[{"input": "a", "output": "b"}]))
    assert isinstance(c1.memory, ComposableMemory)
    assert [getattr(b, "name", "") for b in c1.memory.blocks].count("fewshot") == 1
    assert base.memory is None  # original untouched
    # re-applying replaces the few-shot block (no stacking)
    c2 = apply_candidate(base, Candidate(fewshot_demos=[{"input": "c", "output": "d"}]))
    assert [getattr(b, "name", "") for b in c2.memory.blocks].count("fewshot") == 1


def test_bootstrap_demos_gold_first_no_teacher():
    from fastaiagent.optimize.proposers import bootstrap_demos

    train = [
        {"input": "Capital of France?", "expected_output": "Paris"},
        {"input": "Capital of Japan?", "expected_output": "Tokyo"},
    ]
    # all gold → no agent run; agent.arun is never called (would need an API key).
    demos = asyncio.run(
        bootstrap_demos(
            agent=_agent(),
            train_items=train,
            scorers=["exact_match"],
            judge=None,
            k=5,
            include_favorites=False,
        )
    )
    assert {d["output"] for d in demos} == {"Paris", "Tokyo"}
    assert all(set(d) == {"input", "output"} for d in demos)


# ── Config validation + two-judge default ───────────────────────────────────


def test_config_levers_supported_and_unknown_rejected():
    cfg = OptimizeConfig()
    assert cfg.levers == ("instructions",) and cfg.allow_writable_memory is False
    OptimizeConfig(levers=("instructions", "fewshot", "memory"))  # all three supported (P3)
    with pytest.raises(ValueError, match="not available"):
        OptimizeConfig(levers=("bogus",))


def test_config_rejects_bad_splits():
    with pytest.raises(ValueError, match="splits"):
        OptimizeConfig(splits=(0.5, 0.4, 0.4))


def test_resolve_audit_judge_warns_when_sharing():
    j = LLMJudge(criteria="x", name="sel")
    cfg = OptimizeConfig(selection_judge=j)  # audit None
    with pytest.warns(UserWarning, match="share a judge"):
        assert cfg.resolve_audit_judge() is j


def test_resolve_audit_judge_no_warning_when_distinct():
    sel, aud = LLMJudge(criteria="x", name="sel"), LLMJudge(criteria="y", name="aud")
    cfg = OptimizeConfig(selection_judge=sel, audit_judge=aud)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        assert cfg.resolve_audit_judge() is aud


# ── Proposer: no failures → no proposals (LLM-free early return) ─────────────


def test_proposer_returns_empty_with_no_failures():
    er = _eval_results({"exact_match": [(1.0, True), (1.0, True)]})  # all passed → count 0
    out = asyncio.run(
        propose_prompt_rewrites(current_prompt="p", results=er, llm=None, n=3)  # llm unused
    )
    assert out == []


# ── Report rendering ────────────────────────────────────────────────────────


def _score(score: float, cid: str = "c") -> CandidateScore:
    return CandidateScore(candidate_id=cid, split="dev", score=score, pass_rate=score, n=4)


def test_report_summary_and_dict():
    best_cand = Candidate(system_prompt="better", origin="prompt:rewrite", rationale="added rules")
    report = OptimizationReport(
        agent_name="t",
        baseline=_score(0.50, "base"),
        best=_score(0.75, best_cand.id),
        best_candidate=best_cand,
        trajectory=[
            TrajectoryPoint(0, "baseline", "base", 0.50, True, "baseline"),
            TrajectoryPoint(1, "instructions", best_cand.id, 0.75, True, "added rules"),
            TrajectoryPoint(1, "instructions", "x", 0.40, False, "worse"),
        ],
        accepted=[best_cand.id],
        stopped_reason="patience",
        holdout_baseline=_score(0.50, "base"),
        holdout_best=_score(0.70, best_cand.id),
    )
    s = report.summary()
    assert "baseline" in s and "ACCEPT" in s and "reject" in s and "holdout" in s
    assert report.improved is True
    d = report.to_dict()
    assert d["best"]["score"] == 0.75 and d["accepted"] == [best_cand.id]
    assert len(d["trajectory"]) == 3


def test_report_apply_to_returns_winning_prompt():
    base = _agent("ORIGINAL")
    best_cand = Candidate(system_prompt="WINNER")
    report = OptimizationReport(
        agent_name="t",
        baseline=_score(0.5),
        best=_score(0.8, best_cand.id),
        best_candidate=best_cand,
    )
    assert report.apply_to(base).system_prompt == "WINNER"
    assert base.system_prompt == "ORIGINAL"  # original untouched


# ── P3: memory-fact lever ───────────────────────────────────────────────────


def _seed_facts(tmp_path, monkeypatch, scope="agent", scope_id="a"):
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    from fastaiagent.learn.store import Fact, MemoryStore

    store = MemoryStore()
    ids = store.add_many(
        [
            Fact(scope=scope, scope_id=scope_id, fact="cite sources", confidence=0.9),
            Fact(scope=scope, scope_id=scope_id, fact="under 800 words", confidence=0.5),
            Fact(scope=scope, scope_id=scope_id, fact="prefer primary sources", confidence=0.7),
        ]
    )
    return store, ids


def test_allowlist_store_filters_then_limits(tmp_path, monkeypatch):
    from fastaiagent.optimize.candidate import _AllowlistStore

    store, ids = _seed_facts(tmp_path, monkeypatch)
    allow = _AllowlistStore([ids[0], ids[2]], inner=store)
    assert {f.id for f in allow.list_active("agent", "a")} == {ids[0], ids[2]}
    assert len(allow.list_active("agent", "a", limit=1)) == 1  # limit applied AFTER filter


def test_propose_fact_subsets_ranked_and_empty(tmp_path, monkeypatch):
    from fastaiagent.optimize.proposers import propose_fact_subsets

    store, ids = _seed_facts(tmp_path, monkeypatch)
    subsets = propose_fact_subsets(scope="agent", scope_id="a", n=3, store=store)
    assert [len(s) for s in subsets] == [3, 2, 1]  # full + ablations
    assert subsets[0][0] == ids[0]  # highest confidence (0.9) first
    # empty scope → no subsets (caller skips the lever)
    assert propose_fact_subsets(scope="agent", scope_id="nope", n=3, store=store) == []


def test_resolve_memory_scope_default_and_inherited():
    from fastaiagent.agent.memory import AgentMemory, ComposableMemory
    from fastaiagent.agent.memory_blocks import PersistentFactBlock
    from fastaiagent.optimize.candidate import _resolve_memory_scope

    a = _agent()
    assert _resolve_memory_scope(a) == ("agent", a.name)  # default
    a.memory = ComposableMemory(
        blocks=[PersistentFactBlock(scope="project", scope_id="acme")], primary=AgentMemory()
    )
    assert _resolve_memory_scope(a) == ("project", "acme")  # inherits existing block


def test_apply_candidate_injects_memory_subset(tmp_path, monkeypatch):
    from fastaiagent.agent.memory import ComposableMemory

    _store, ids = _seed_facts(tmp_path, monkeypatch, scope_id="t")  # agent name is "t"
    base = _agent("P")
    cand = apply_candidate(base, Candidate(fact_ids=[ids[0]]))
    assert isinstance(cand.memory, ComposableMemory)
    rendered = " ".join(m.content for b in cand.memory.blocks for m in b.render(""))
    assert "cite sources" in rendered and "800 words" not in rendered
    assert base.memory is None  # original untouched


def test_memory_lever_never_mutates_store(tmp_path, monkeypatch):
    # Selection is run-local: applying fact_ids candidates + proposing subsets must
    # never create/edit/delete/supersede facts — the audit chain stays intact.
    from fastaiagent.optimize.proposers import propose_fact_subsets

    store, ids = _seed_facts(tmp_path, monkeypatch, scope_id="t")
    before = [(f.id, f.fact, f.confidence, f.superseded_by) for f in store.list_all()]
    base = _agent("P")
    for subset in ([ids[0]], ids, []):
        apply_candidate(base, Candidate(fact_ids=subset))
    propose_fact_subsets(scope="agent", scope_id="t", n=3, store=store)
    after = [(f.id, f.fact, f.confidence, f.superseded_by) for f in store.list_all()]
    assert before == after  # store + audit chain unchanged


# ── 1.82.0: a Memory stays a Memory through optimize ────────────────────────


@dataclass
class _St:
    user_id: str


def _as(uid: str) -> RunContext:
    return RunContext(state=_St(user_id=uid))


def _prompt(call: dict) -> str:
    return " ".join(m.content or "" for m in call["messages"] if isinstance(m.content, str))


def _memory_agent(mem) -> tuple[Agent, TestModel]:
    model = TestModel(response="noted")
    return Agent(name="support", system_prompt="P", llm=model, memory=mem), model


def test_optimized_per_user_memory_keeps_users_apart(tmp_path, monkeypatch):
    """``apply_to`` turned a per-user Memory into one ComposableMemory, so every
    user of the optimized agent shared one window."""
    from fastaiagent.agent.memory_simple import Memory
    from fastaiagent.learn.store import MemoryStore

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    mem = Memory(location=MemoryStore(), user_id=lambda ctx: ctx.state.user_id, window=4)
    base, model = _memory_agent(mem)
    # What OptimizationReport.apply_to does.
    optimized = apply_candidate(base, Candidate(system_prompt="P2"), allow_writable_memory=True)

    optimized.run("alice-private-message", context=_as("alice"))
    optimized.run("bob here", context=_as("bob"))
    assert "alice-private-message" not in _prompt(model.calls[-1])
    assert isinstance(optimized.memory, Memory)
    assert optimized.memory is not mem
    for i in range(4):
        optimized.run(f"alice turn {i}", context=_as("alice"))
    assert len(optimized.memory.for_user("alice").messages) <= 4  # window= survives
    assert mem.for_user("alice").messages == []  # the original is untouched


def test_optimize_refuses_a_learning_per_user_memory_up_front(tmp_path, monkeypatch):
    """A per-user Memory hid learn= behind the anonymous caller's blocks, so the
    isolation guard never saw it and the deployed agent silently lost it."""
    from fastaiagent.agent.memory_blocks import MemoryIsolationError
    from fastaiagent.agent.memory_simple import Memory
    from fastaiagent.learn.store import MemoryStore

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    mem = Memory(location=MemoryStore(), user_id=lambda ctx: ctx.state.user_id, learn=TestModel())
    with pytest.raises(MemoryIsolationError, match="learn="):
        _clone_memory_blocks(mem)
    with pytest.warns(UserWarning, match="learn="):
        shared = _clone_memory_blocks(mem, allow_writable_memory=True)
    assert isinstance(shared, Memory)
    kinds = [type(b).__name__ for b in shared.for_user("alice").blocks]
    assert "FactExtractionBlock" in kinds  # learn= kept, not dropped


def test_auto_recall_is_isolated_per_candidate(tmp_path, monkeypatch):
    """recall="auto" builds a fresh in-process index per user of each copy, so
    it is isolated; only a VectorStore you pass is shared and refused."""
    from fastaiagent.agent.memory_blocks import MemoryIsolationError
    from fastaiagent.agent.memory_simple import Memory
    from fastaiagent.kb.backends.faiss import FaissVectorStore
    from fastaiagent.learn.store import MemoryStore

    pytest.importorskip("faiss")
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))

    class _Embedder:
        def embed(self, texts):
            return [[float(len(t)), 1.0, 0.0] for t in texts]

    kw = dict(location=MemoryStore(), user_id=lambda ctx: ctx.state.user_id, embedder=_Embedder())
    assert isinstance(_clone_memory_blocks(Memory(recall="auto", **kw)), Memory)
    shared_store = FaissVectorStore(dimension=3, index_type="flat")
    with pytest.raises(MemoryIsolationError, match="recall="):
        _clone_memory_blocks(Memory(recall=shared_store, **kw))


def test_memory_levers_reach_a_resolved_user(tmp_path, monkeypatch):
    """The fact and few-shot levers apply to every user, and each user keeps
    their own facts."""
    from fastaiagent.agent.memory_simple import Memory
    from fastaiagent.learn.store import Fact

    store, ids = _seed_facts(tmp_path, monkeypatch, scope_id="support")
    store.add(Fact(scope="user", scope_id="alice", fact="alice likes tea"))
    mem = Memory(location=store, agent_id="support", user_id=lambda ctx: ctx.state.user_id)
    base, model = _memory_agent(mem)
    cand = Candidate(fact_ids=[ids[0]], fewshot_demos=[{"input": "demo-in", "output": "demo-out"}])
    optimized = apply_candidate(base, cand)

    optimized.run("hi", context=_as("alice"))
    seen = _prompt(model.calls[-1])
    assert "cite sources" in seen and "800 words" not in seen  # the fact subset
    assert "demo-in" in seen  # the few-shot demos
    assert "alice likes tea" in seen  # the user's own facts survive


def test_static_user_memory_keeps_its_window_and_store_verbs(tmp_path, monkeypatch):
    from fastaiagent.agent.memory_simple import Memory
    from fastaiagent.learn.store import MemoryStore

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    mem = Memory(location=MemoryStore(), user_id="alice", window=2)
    mem.persist("alice likes tea", tier="user", id="alice")
    base, _model = _memory_agent(mem)
    optimized = apply_candidate(base, Candidate(system_prompt="P2"))

    assert isinstance(optimized.memory, Memory)
    for i in range(3):
        optimized.run(f"turn {i}")
    assert len(optimized.memory.messages) <= 2
    assert [f.fact for f in optimized.memory.retrieve(tier="user", id="alice")] == [
        "alice likes tea"
    ]


def test_resolve_memory_scope_for_a_memory(tmp_path, monkeypatch):
    """The memory lever replaces the global fact block, so it selects among the
    agent's global facts — never a user's."""
    from fastaiagent.agent.memory_simple import Memory
    from fastaiagent.learn.store import MemoryStore
    from fastaiagent.optimize.candidate import _resolve_memory_scope

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    a, _ = _memory_agent(Memory(location=MemoryStore(), user_id="alice", agent_id="kb"))
    assert _resolve_memory_scope(a) == ("agent", "kb")
    b, _ = _memory_agent(Memory(location=MemoryStore(), user_id="alice"))
    assert _resolve_memory_scope(b) == ("agent", "support")


# ── 1.82.0: the memory lever reads the agent's own fact store ───────────────


def _own_store_agent(tmp_path, monkeypatch):
    """An agent whose Memory keeps facts in its own store under a project — and a
    default local.db that holds none of them."""
    from fastaiagent.agent.memory_simple import Memory
    from fastaiagent.learn.store import Fact, MemoryStore

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "default-local.db"))
    store = MemoryStore(db_path=str(tmp_path / "agent-facts.db"))
    ids = [
        store.add(Fact(scope="agent", scope_id="kb", fact=text, confidence=c, project_id="acme"))
        for text, c in (("cite sources", 0.9), ("under 800 words", 0.5), ("prefer primary", 0.7))
    ]
    mem = Memory(
        location=store, agent_id="kb", project_id="acme", user_id=lambda ctx: ctx.state.user_id
    )
    agent, model = _memory_agent(mem)
    return agent, model, store, ids


def test_memory_lever_source_is_the_agents_own_store(tmp_path, monkeypatch):
    from fastaiagent.optimize.candidate import _resolve_memory_source

    agent, _model, store, _ids = _own_store_agent(tmp_path, monkeypatch)
    src = _resolve_memory_source(agent)
    assert (src.scope, src.scope_id, src.project_id) == ("agent", "kb", "acme")
    assert [f.fact for f in src.store.list_active("agent", "kb", "acme")] == [
        f.fact for f in store.list_active("agent", "kb", "acme")
    ]


def test_selected_facts_come_from_the_agents_own_store(tmp_path, monkeypatch):
    """The allowlist read the default local.db, so a fact selected from the
    agent's store (or under its project) was never injected."""
    agent, model, _store, ids = _own_store_agent(tmp_path, monkeypatch)
    optimized = apply_candidate(agent, Candidate(fact_ids=[ids[0]]))
    optimized.run("hi", context=_as("alice"))
    seen = _prompt(model.calls[-1])
    assert "cite sources" in seen and "800 words" not in seen


def test_propose_fact_subsets_honours_project_id(tmp_path, monkeypatch):
    from fastaiagent.optimize.proposers import propose_fact_subsets

    _agent_, _model, store, ids = _own_store_agent(tmp_path, monkeypatch)
    subsets = propose_fact_subsets(
        scope="agent", scope_id="kb", n=3, store=store, project_id="acme"
    )
    assert subsets and subsets[0][0] == ids[0]


def test_memory_lever_runs_over_the_agents_own_store(tmp_path, monkeypatch):
    """optimize() looked for facts in the default local.db, found none, and
    skipped the memory lever for any agent that keeps its facts elsewhere."""
    from fastaiagent.optimize import optimize

    agent, _model, _store, _ids = _own_store_agent(tmp_path, monkeypatch)
    cases = [{"input": f"q{i}", "expected_output": "noted"} for i in range(6)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = optimize(
            agent,
            cases,
            scorers=["contains"],
            config=OptimizeConfig(levers=("memory",), max_iterations=1),
            persist=False,
        )
    memory_points = [p for p in report.trajectory if p.lever == "memory"]
    assert memory_points and not any(p.skipped for p in memory_points)


def test_memory_lever_source_follows_a_fact_blocks_store(tmp_path, monkeypatch):
    from fastaiagent.agent.memory import AgentMemory, ComposableMemory
    from fastaiagent.agent.memory_blocks import PersistentFactBlock
    from fastaiagent.learn.store import Fact, MemoryStore
    from fastaiagent.optimize.candidate import _resolve_memory_source

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "default-local.db"))
    store = MemoryStore(db_path=str(tmp_path / "block-facts.db"))
    fid = store.add(Fact(scope="project", scope_id="docs", fact="be brief", project_id="t1"))
    block = PersistentFactBlock(scope="project", scope_id="docs", project_id="t1", store=store)
    agent, model = _memory_agent(ComposableMemory(blocks=[block], primary=AgentMemory()))

    src = _resolve_memory_source(agent)
    assert (src.scope, src.scope_id, src.project_id, src.store) == ("project", "docs", "t1", store)
    apply_candidate(agent, Candidate(fact_ids=[fid])).run("hi")
    assert "be brief" in _prompt(model.calls[-1])


# ── 1.82.0: the memory lever and a per-user fact block ──────────────────────


def _per_user_block_agent(tmp_path, monkeypatch):
    """A ComposableMemory whose fact block resolves the user per run, plus shared
    facts at the agent's default scope."""
    from fastaiagent.agent.memory import AgentMemory, ComposableMemory
    from fastaiagent.agent.memory_blocks import PersistentFactBlock
    from fastaiagent.learn.store import Fact, MemoryStore

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    store = MemoryStore()
    store.add(Fact(scope="user", scope_id="alice", fact="alice likes tea"))
    ids = [
        store.add(Fact(scope="agent", scope_id="support", fact=text, confidence=c))
        for text, c in (("cite sources", 0.9), ("under 800 words", 0.5))
    ]
    block = PersistentFactBlock(scope="user", scope_id=lambda ctx: ctx.state.user_id, store=store)
    agent, model = _memory_agent(ComposableMemory(blocks=[block], primary=AgentMemory()))
    return agent, model, ids


def test_memory_lever_with_a_per_user_fact_block_does_not_crash(tmp_path, monkeypatch):
    """The lever took the per-user block's resolver function for a scope id and
    handed it to the store: optimize() raised ProgrammingError."""
    from fastaiagent.optimize import optimize
    from fastaiagent.optimize.candidate import _resolve_memory_source

    agent, _model, _ids = _per_user_block_agent(tmp_path, monkeypatch)
    src = _resolve_memory_source(agent)
    assert (src.scope, src.scope_id) == ("agent", "support")  # the shared facts
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = optimize(
            agent,
            [{"input": f"q{i}", "expected_output": "noted"} for i in range(6)],
            scorers=["contains"],
            config=OptimizeConfig(levers=("memory",), max_iterations=1),
            persist=False,
        )
    assert [p for p in report.trajectory if p.lever == "memory" and not p.skipped]


def test_the_optimized_agent_keeps_each_users_own_facts(tmp_path, monkeypatch):
    """Injecting the selected facts replaced every block named
    "persistent_facts" — including the per-user one, so users lost their facts."""
    agent, model, ids = _per_user_block_agent(tmp_path, monkeypatch)
    optimized = apply_candidate(agent, Candidate(fact_ids=[ids[0]]))
    names = [b.name for b in optimized.memory.blocks]
    assert names == ["persistent_facts", "persistent_facts.optimized"]  # distinct labels
    optimized.run("hi", context=_as("alice"))
    seen = _prompt(model.calls[-1])
    assert "alice likes tea" in seen  # the user's own facts
    assert "cite sources" in seen and "800 words" not in seen  # the selected shared facts
    again = apply_candidate(optimized, Candidate(fact_ids=[ids[1]]))  # re-optimize: no stacking
    assert [b.name for b in again.memory.blocks] == names
