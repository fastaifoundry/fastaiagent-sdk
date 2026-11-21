"""A chain loop does what its edge says (1.105.0).

Four defects in the cyclic-edge path of ``chain/executor.py``, all present since 0.1 and all
found when a coding assistant built a "draft, review, retry at most twice" workflow and wrote its
own attempt cap because the SDK's did not hold:

* ``max_iterations`` above 1 was never enforced. A loop pass re-enters ``execute_chain`` and each
  call started its own counters, so every frame counted to 1. A reviewer that never passed ran
  about 980 passes, then ``RecursionError``.
* A node after the loop ran once per pass (three reviews, three sends).
* The chain's middleware was not applied inside a loop pass.
* The exit condition read state from before the node ran, and a pass did not see the results of
  nodes that ran before the loop.

No mocks: real chains, scripted models, call counters.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fastaiagent import Agent, AgentMiddleware, Chain
from fastaiagent._internal.errors import ChainCycleError
from fastaiagent.testing import FunctionModel

PASS = "{{node_results.review.output}} startswith PASS"


class Calls(dict):
    def __init__(self) -> None:
        super().__init__(draft=0, review=0, send=0)


def _loop(
    calls: Calls,
    pass_on: int,
    *,
    cap: int | None,
    exit_condition: str | None = PASS,
    middleware: list[Any] | None = None,
    after: bool = True,
) -> Chain:
    """draft -> review -> (send), with review looping back to draft."""

    def model(name: str, reply: Callable[[], str]) -> FunctionModel:
        def fn(messages: Any) -> str:
            calls[name] += 1
            return reply()

        return FunctionModel(fn)

    chain = Chain("loop", checkpoint_enabled=False, middleware=middleware or [])
    chain.add_node("draft", agent=Agent(name="d", llm=model("draft", lambda: "a draft")))
    chain.add_node(
        "review",
        agent=Agent(
            name="r",
            llm=model("review", lambda: "PASS" if calls["review"] >= pass_on else "FAIL"),
        ),
    )
    chain.connect("draft", "review")
    if after:
        chain.add_node("send", agent=Agent(name="s", llm=model("send", lambda: "sent")))
        chain.connect("review", "send")
    kw: dict[str, Any] = {}
    if cap is not None:
        kw["max_iterations"] = cap
    if exit_condition is not None:
        kw["exit_condition"] = exit_condition
    chain.connect("review", "draft", **kw)
    return chain


# ── the cap ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cap", [1, 2, 3, 5])
def test_a_loop_that_never_exits_stops_at_max_iterations(isolated_local_db: Path, cap: int) -> None:
    calls = Calls()
    with pytest.raises(ChainCycleError, match=rf"max_iterations \({cap}\)"):
        _loop(calls, pass_on=10_000, cap=cap).execute({"input": "x"}, trace=False)
    assert (calls["draft"], calls["review"]) == (cap, cap)
    assert calls["send"] == 0  # the chain failed; nothing after the loop ran


def test_the_cap_holds_without_an_exit_condition(isolated_local_db: Path) -> None:
    calls = Calls()
    with pytest.raises(ChainCycleError):
        _loop(calls, pass_on=10_000, cap=3, exit_condition=None).execute(
            {"input": "x"}, trace=False
        )
    assert calls["review"] == 3


def test_a_loop_with_only_an_exit_condition_gets_the_default_cap_of_ten(
    isolated_local_db: Path,
) -> None:
    calls = Calls()
    with pytest.raises(ChainCycleError, match=r"max_iterations \(10\)"):
        _loop(calls, pass_on=10_000, cap=None).execute({"input": "x"}, trace=False)
    assert calls["review"] == 10


# ── what runs, and how often ─────────────────────────────────────────────────


@pytest.mark.parametrize("pass_on", [1, 2, 3, 4])
def test_the_node_after_the_loop_runs_once_whatever_the_pass_count(
    isolated_local_db: Path, pass_on: int
) -> None:
    calls = Calls()
    result = _loop(calls, pass_on=pass_on, cap=5).execute({"input": "x"}, trace=False)
    assert result.status == "completed"
    assert (calls["draft"], calls["review"]) == (pass_on, pass_on)
    assert calls["send"] == 1
    assert result.node_results["send"]["output"] == "sent"


def test_exiting_on_the_last_allowed_pass_is_not_an_error(isolated_local_db: Path) -> None:
    calls = Calls()
    result = _loop(calls, pass_on=3, cap=3).execute({"input": "x"}, trace=False)
    assert result.status == "completed"
    assert calls["review"] == 3 and calls["send"] == 1


def test_a_loop_at_the_end_of_the_chain_returns_the_last_pass(isolated_local_db: Path) -> None:
    calls = Calls()
    result = _loop(calls, pass_on=2, cap=5, after=False).execute({"input": "x"}, trace=False)
    assert result.status == "completed"
    assert calls["review"] == 2
    assert result.node_results["review"]["output"] == "PASS"


# ── middleware ───────────────────────────────────────────────────────────────


class CountRuns(AgentMiddleware):
    """Counts agent runs in its own per-run state, and on the instance."""

    def __init__(self) -> None:
        self.total = 0
        self.state_counts: list[int] = []

    async def before_run(self, ctx: Any, input: Any) -> Any:
        self.total += 1
        ctx.state["n"] = ctx.state.get("n", 0) + 1
        self.state_counts.append(ctx.state["n"])
        return input


def test_chain_middleware_runs_on_every_pass(isolated_local_db: Path) -> None:
    calls, mw = Calls(), CountRuns()
    _loop(calls, pass_on=3, cap=5, middleware=[mw]).execute({"input": "x"}, trace=False)
    assert mw.total == calls["draft"] + calls["review"] + calls["send"] == 7


def test_middleware_state_spans_the_loop_instead_of_resetting_each_pass(
    isolated_local_db: Path,
) -> None:
    """A budget that reset on every pass would never be reached."""
    calls, mw = Calls(), CountRuns()
    _loop(calls, pass_on=3, cap=5, middleware=[mw]).execute({"input": "x"}, trace=False)
    assert mw.state_counts == [1, 2, 3, 4, 5, 6, 7]


# ── what a pass can see ──────────────────────────────────────────────────────


def test_an_exit_condition_on_state_sees_what_the_node_just_wrote(isolated_local_db: Path) -> None:
    """``review`` writes its verdict to state via output_key; the loop must exit on the pass
    that wrote PASS, not one pass later."""
    calls = Calls()

    def model(name: str, reply: Callable[[], str]) -> FunctionModel:
        def fn(messages: Any) -> str:
            calls[name] += 1
            return reply()

        return FunctionModel(fn)

    chain = Chain("state-exit", checkpoint_enabled=False)
    chain.add_node("draft", agent=Agent(name="d", llm=model("draft", lambda: "a draft")))
    chain.add_node(
        "review",
        agent=Agent(
            name="r", llm=model("review", lambda: "PASS" if calls["review"] >= 2 else "FAIL")
        ),
        output_key="verdict",
    )
    chain.connect("draft", "review")
    chain.connect("review", "draft", max_iterations=5, exit_condition="{{state.verdict}} == PASS")
    result = chain.execute({"input": "x"}, trace=False)
    assert result.status == "completed"
    assert calls["review"] == 2


def test_a_pass_can_route_on_a_node_that_ran_before_the_loop(isolated_local_db: Path) -> None:
    """``send`` is routed on ``intake``'s result. The final routing happens inside a loop pass,
    which used to start with no earlier node results, so ``send`` never ran."""
    calls = Calls()

    def model(name: str, reply: Callable[[], str]) -> FunctionModel:
        def fn(messages: Any) -> str:
            calls[name] = calls.get(name, 0) + 1
            return reply()

        return FunctionModel(fn)

    chain = Chain("see-earlier", checkpoint_enabled=False)
    chain.add_node("intake", agent=Agent(name="i", llm=model("intake", lambda: "go")))
    chain.add_node("draft", agent=Agent(name="d", llm=model("draft", lambda: "a draft")))
    chain.add_node(
        "review",
        agent=Agent(
            name="r", llm=model("review", lambda: "PASS" if calls["review"] >= 2 else "FAIL")
        ),
    )
    chain.add_node("send", agent=Agent(name="s", llm=model("send", lambda: "sent")))
    chain.connect("intake", "draft")
    chain.connect("draft", "review")
    chain.connect("review", "send", condition="{{node_results.intake.output}} == go")
    chain.connect("review", "draft", max_iterations=5, exit_condition=PASS)
    result = chain.execute({"input": "x"}, trace=False)
    assert result.status == "completed"
    assert (calls["intake"], calls["review"], calls["send"]) == (1, 2, 1)


# ── a loop that is not marked as one ─────────────────────────────────────────


def test_an_exit_condition_alone_makes_a_loop_with_the_default_cap(isolated_local_db: Path) -> None:
    """It used to be dropped: the edge was an ordinary one, the two nodes sat on an unordered
    loop, and the chain "completed" having run nothing."""
    calls = Calls()
    chain = _loop(calls, pass_on=2, cap=None, after=False)
    (back,) = [e for e in chain.edges if e.source == "review"]
    assert back.is_cyclic and back.cycle_config["max_iterations"] == 10
    assert chain.execute({"input": "x"}, trace=False).status == "completed"
    assert calls["review"] == 2


def test_validate_reports_a_loop_drawn_with_ordinary_edges() -> None:
    chain = Chain("unmarked", checkpoint_enabled=False)
    for name in ("a", "b", "c"):
        chain.add_node(name, agent=Agent(name=name, llm=FunctionModel(lambda m: "x")))
    chain.connect("a", "b").connect("b", "c").connect("c", "b")
    (error,) = chain.validate()
    assert "['b', 'c'] form a loop through ordinary edges" in error
    assert "max_iterations" in error


def test_validate_accepts_a_marked_loop() -> None:
    assert _loop(Calls(), pass_on=1, cap=3).validate() == []
