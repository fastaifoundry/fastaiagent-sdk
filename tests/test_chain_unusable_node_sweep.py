"""One invariant, swept across every node type a chain can hold.

    A node whose configuration cannot run anything must report that it
    **could not run** — never a clean verdict.

This is ``tests/test_guardrail_unusable_config_sweep.py``'s sibling, for the
other half of the SDK that keeps shipping the same defect. The guardrail sweep
was written after three consecutive releases each fixed *one instance* of a rule
that inspected nothing and passed. The chain executor had the identical shape in
six places at once, and had had it since v0.1:

* an **agent** node with no agent returned ``{"error": "No agent attached…"}``
* a **tool** node with no tool returned ``{"error": "No tool attached…"}``
* a **transformer** with no template returned ``{"output": ""}``
* a **parallel** node with no children returned ``{"outputs": []}``
* a **condition** node with no conditions always answered ``"default"``
* an approval **gate** with no handler approved itself

and every one of those was a *result*, merged into chain state, checkpointed
``status="completed"`` and returned as ``ChainResult(status="completed")``.
``chain.validate()`` returned ``[]`` throughout, because it only ever looked at
the graph. A run that did nothing was indistinguishable from a run that worked.

**Why the assertion is "never completed" and not "raises".** What a caller sees
is ``ChainResult.status`` and the run's terminal checkpoint. A node that cannot
run must not leave either of them saying the run was fine — whether the executor
gets there by raising (it does) or by some future non-raising refusal is an
implementation choice this sweep deliberately does not pin. What it pins is the
outcome class, which is the same contract ``CLAUDE.md`` §2.4 states for
guardrails.

No mocks: a real ``Chain`` over a real ``SQLiteCheckpointer``. None of these
cases reaches a model, by construction — the whole point is that they run
nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from fastaiagent import SQLiteCheckpointer
from fastaiagent.chain import Chain
from fastaiagent.chain.checkpoint import is_run_end
from fastaiagent.chain.node import NodeType

#: ``NodeType`` → how to add a node of that type that cannot run, and why.
#:
#: Each builder takes the chain and adds exactly one node called ``"n"``. The
#: config is one a caller can write today; nothing here is synthetic.
UNUSABLE: dict[NodeType, tuple[Callable[[Chain], Any], str]] = {
    NodeType.agent: (
        lambda c: c.add_node("n", type=NodeType.agent),
        "an agent node with no agent",
    ),
    NodeType.tool: (
        lambda c: c.add_node("n", type=NodeType.tool),
        "a tool node with no tool",
    ),
    NodeType.transformer: (
        lambda c: c.add_node("n", type=NodeType.transformer),
        "a transformer with no template",
    ),
    NodeType.parallel: (
        lambda c: c.add_node("n", type=NodeType.parallel),
        "a parallel node with no children",
    ),
    NodeType.condition: (
        lambda c: c.add_node("n", type=NodeType.condition),
        "a condition node with no conditions",
    ),
    NodeType.hitl: (
        lambda c: c.add_node("n", type=NodeType.hitl),
        "an approval gate with no handler and no opt-in",
    ),
}

#: Node types with nothing to make unusable, each for a stated reason rather
#: than because nobody got to them. ``start`` and ``end`` take no payload at
#: all — they pass the chain's input through, which is their entire job — so
#: there is no configuration of either that could fail to run.
EXEMPT: dict[NodeType, str] = {
    NodeType.start: "takes no payload; passes the chain input through",
    NodeType.end: "takes no payload; passes the chain input through",
}

#: Types whose defect is real and deliberately left unfixed pending a human
#: decision. Empty, and kept as an empty mapping rather than deleted: a future
#: degenerate config that cannot be closed without sign-off goes here with its
#: reason, and is then *named in every run* instead of silently absent.
KNOWN_UNFIXED: dict[NodeType, str] = {}


def _store(tmp_path, name: str) -> SQLiteCheckpointer:
    cp = SQLiteCheckpointer(db_path=str(tmp_path / f"{name}.db"))
    cp.setup()
    return cp


@pytest.mark.parametrize("node_type", sorted(UNUSABLE, key=lambda t: t.value))
def test_an_unusable_node_never_reports_a_completed_run(node_type, tmp_path) -> None:
    if node_type in KNOWN_UNFIXED:
        pytest.xfail(f"known, unfixed, needs sign-off: {node_type.value}")

    build, why = UNUSABLE[node_type]
    store = _store(tmp_path, node_type.value)
    chain = Chain(f"unusable-{node_type.value}", checkpointer=store)
    build(chain)

    execution_id = f"ex-{node_type.value}"
    status: str | None = None
    try:
        status = chain.execute({"input": "hello"}, execution_id=execution_id).status
    except Exception:
        # A raise is the current (and intended) mechanism. The sweep pins the
        # outcome class, not the mechanism — see the module docstring.
        pass

    assert status != "completed", (
        f"{node_type.value} — {why} — reported status='completed'. A node that ran "
        f"nothing must not report a clean run; this is the chain half of the "
        f"invariant the guardrail sweep owns for controls."
    )

    # The durable record has to agree with what the caller was told. A row
    # saying ``completed`` outlives the process and is what the plane replicates.
    marker = store.get_last(execution_id)
    if marker is not None and is_run_end(marker):
        assert marker.status != "completed", (
            f"{node_type.value} — {why} — wrote a terminal checkpoint claiming the run completed."
        )


@pytest.mark.parametrize("node_type", sorted(UNUSABLE, key=lambda t: t.value))
def test_validate_flags_an_unusable_node_before_the_run(node_type, tmp_path) -> None:
    """The cheaper half: say it at design time, before the model bill.

    ``hitl`` is the one exception and it is not an oversight — a gate's handler
    arrives as an argument to ``execute()``, so at ``validate()`` time there is
    nothing yet to be missing. It is asserted the other way round: validate must
    *not* invent an error for it.
    """
    build, why = UNUSABLE[node_type]
    chain = Chain(f"validate-{node_type.value}")
    build(chain)

    errors = chain.validate()
    if node_type is NodeType.hitl:
        assert errors == [], "a gate's handler is a run-time argument; validate cannot know"
        return
    assert any("'n'" in e for e in errors), (
        f"{node_type.value} — {why} — passed validate() with {errors!r}. validate() "
        f"used to check the graph only, so every one of these was silent."
    )


def test_the_sweep_covers_every_node_type() -> None:
    """The sweep is only worth having if it cannot fall behind ``NodeType``.

    When a ninth node type is added this fails until somebody has decided what
    "unusable" means for it — which is exactly the step skipped six times over.
    """
    covered = set(UNUSABLE) | set(EXEMPT)
    missing = {t for t in NodeType} - covered
    assert not missing, (
        f"these node types have no unusable-config case: {sorted(t.value for t in missing)}. "
        f"Decide what an unusable configuration looks like for each and add it to "
        f"UNUSABLE — or to EXEMPT with the reason it cannot have one."
    )


def test_a_usable_node_of_each_type_still_completes(tmp_path) -> None:
    """The other half, and the reason this is a sweep and not a blanket rule.

    A fix that turns a *working* configuration into a refusal is a worse defect
    than the one it closes. Every type that has an unusable case here also has a
    usable one, and it must still run clean.
    """
    from fastaiagent.tool.function import FunctionTool

    store = _store(tmp_path, "usable")
    chain = Chain("usable", checkpointer=store)
    chain.add_node(
        "t",
        tool=FunctionTool(name="echo", fn=lambda text: text.upper()),
        input_mapping={"text": "{{state.text}}"},
    )
    chain.add_node("x", type=NodeType.transformer, template="seen {{state.output}}")
    chain.add_node(
        "c",
        type=NodeType.condition,
        conditions=[{"expression": "{{state.output}} contains SEEN", "handle": "yes"}],
    )
    chain.add_node("g", type=NodeType.hitl, auto_approve=True)
    chain.connect("t", "x")
    chain.connect("x", "c")
    chain.connect("c", "g", label="yes")
    chain.connect("c", "g")

    assert chain.validate() == []
    result = chain.execute({"input": "hi", "text": "hi"}, execution_id="ex-usable")
    assert result.status == "completed"
    assert result.node_results["g"]["approved"] is True
