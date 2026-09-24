"""End-to-end quality gate — Human-in-the-Loop (HITL) chain nodes.

HITL nodes let a chain pause for human approval before continuing.
They are used for high-stakes workflows where automated output must
be reviewed by a human: customer-service replies, content moderation,
compliance-gated decisions, etc.

Two SDK moving parts this gate covers:

1. ``NodeType.hitl`` — a chain node type that represents "pause here
   for approval". It does not call an agent or a tool; its result
   dict carries an ``approved`` field and, when the node opted in with
   ``auto_approve=True``, an ``Auto-approved (auto_approve=True)``
   message. Since 1.67.0 a gate with **neither** a handler nor that
   opt-in refuses the run rather than approving itself.

2. ``hitl_handler`` — a callable passed to ``Chain.execute()`` /
   ``Chain.aexecute()``. Called with ``(node, context, state)`` each
   time execution hits a HITL node; whatever it returns becomes the
   ``approved`` value on the node's result dict.

Six sub-tests, all self-contained (no LLM, no network, no platform):

1. A gate with no handler and no opt-in refuses; a gate that opted
   in with ``auto_approve=True`` completes and carries the documented
   message.
2. Custom handler is invoked with the expected three arguments
   ``(node, context, state)`` and ``context`` exposes the expected
   keys (``input``, ``state``, ``node_results``).
3. Handler return value lands on the node result — True and False
   both round-trip correctly to ``result.node_results[hitl_id]["approved"]``.
4. Handler can read prior node outputs via ``context["node_results"]``
   — proves the handler sees the DRAFT node's result before deciding.
5. Approved flow continues past the HITL node — a 3-node chain
   (draft -> review -> finalize) runs all three nodes when the
   handler returns True.
6. **A rejection halts the chain (since 1.77.0, audit M9).** The gate
   records ``approved=False``, nothing after it runs, and the result has
   ``status="rejected"``. Until 1.77.0 the chain kept running; this gate
   pinned that, and flagged the change when it was made deliberately.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


def _draft_fn() -> dict[str, Any]:
    """Simulates a draft-producing step. Returns a canned draft string."""
    return {"draft_text": "Dear customer, your issue will be resolved shortly."}


def _finalize_fn(draft: str) -> dict[str, Any]:
    """Simulates a finalize step that reads the draft via input_mapping."""
    return {"finalized": f"FINALIZED: {draft}"}


def _build_chain_draft_review_finalize():
    """Build a 3-node chain: draft (tool) -> review (HITL) -> finalize (tool)."""
    from fastaiagent import Chain, FunctionTool
    from fastaiagent.chain.node import NodeType

    chain = Chain("hitl-gate-chain", checkpoint_enabled=False)
    chain.add_node(
        "draft",
        tool=FunctionTool(name="draft_tool", fn=_draft_fn),
        type=NodeType.tool,
    )
    chain.add_node("review", type=NodeType.hitl)
    chain.add_node(
        "finalize",
        tool=FunctionTool(name="finalize_tool", fn=_finalize_fn),
        type=NodeType.tool,
        input_mapping={"draft": "{{state.output.draft_text}}"},
    )
    chain.connect("draft", "review")
    chain.connect("review", "finalize")
    return chain


class TestHITLGate:
    """HITL contract: auto-approve, handler args, return value, continuation."""

    # ── 1. Auto-approve path ────────────────────────────────────────────

    def test_01_auto_approve_when_asked_for(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        import pytest

        from fastaiagent import Chain
        from fastaiagent._internal.errors import ChainError
        from fastaiagent.chain.node import NodeType

        # A gate nobody can answer must not answer itself (1.67.0). Before that,
        # forgetting ``hitl_handler=`` on a call that had it in development was
        # enough to wave the gate through and still report ``completed``.
        unconfigured = Chain("hitl-no-handler-gate", checkpoint_enabled=False)
        unconfigured.add_node("review", type=NodeType.hitl)
        with pytest.raises(ChainError, match="no handler"):
            unconfigured.execute({"message": "go"})

        chain = Chain("hitl-auto-approve-gate", checkpoint_enabled=False)
        chain.add_node("review", type=NodeType.hitl, auto_approve=True)

        # The opt-in is on the node, so it is visible in the chain's definition.
        result = chain.execute({"message": "go"})

        assert result is not None, "auto-approve chain returned None"
        assert "review" in result.node_results, (
            f"HITL node did not record a result: {result.node_results}"
        )
        review = result.node_results["review"]
        assert isinstance(review, dict), f"review result not a dict: {review!r}"
        assert review.get("approved") is True, (
            f"auto-approve did not set approved=True: {review}"
        )
        assert "Auto-approved" in (review.get("message") or ""), (
            f"auto-approve message missing from result: {review}"
        )

    # ── 2. Custom handler receives expected args ─────────────────────────

    def test_02_handler_called_with_node_context_state(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        from fastaiagent import Chain
        from fastaiagent.chain.node import NodeType

        captured: dict[str, Any] = {}

        def handler(node, context, state):
            captured["node"] = node
            captured["context"] = context
            captured["state"] = state
            captured["call_count"] = captured.get("call_count", 0) + 1
            return True

        chain = Chain("hitl-handler-args-gate", checkpoint_enabled=False)
        chain.add_node("review", type=NodeType.hitl)
        chain.execute({"message": "hello"}, hitl_handler=handler)

        assert captured.get("call_count") == 1, (
            f"handler should have been called exactly once, got "
            f"{captured.get('call_count')}"
        )
        # node should be the NodeConfig for the HITL node.
        assert captured["node"] is not None, "handler saw node=None"
        assert getattr(captured["node"], "id", None) == "review", (
            f"handler saw wrong node: {captured['node']}"
        )
        # context must expose the documented keys.
        ctx = captured["context"]
        assert isinstance(ctx, dict), f"context is not a dict: {type(ctx)}"
        for key in ("input", "state", "node_results"):
            assert key in ctx, (
                f"handler context missing documented key {key!r}: "
                f"{list(ctx.keys())}"
            )
        # state is the ChainState object, not a dict.
        assert captured["state"] is not None

    # ── 3. Handler return value round-trips to node result ──────────────

    def test_03_handler_return_true_sets_approved_true(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        from fastaiagent import Chain
        from fastaiagent.chain.node import NodeType

        chain = Chain("hitl-true-gate", checkpoint_enabled=False)
        chain.add_node("review", type=NodeType.hitl)
        result = chain.execute(
            {"message": "go"}, hitl_handler=lambda n, c, s: True
        )
        assert result.node_results["review"]["approved"] is True

    def test_04_handler_return_false_sets_approved_false(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        from fastaiagent import Chain
        from fastaiagent.chain.node import NodeType

        chain = Chain("hitl-false-gate", checkpoint_enabled=False)
        chain.add_node("review", type=NodeType.hitl)
        result = chain.execute(
            {"message": "go"}, hitl_handler=lambda n, c, s: False
        )
        assert result.node_results["review"]["approved"] is False

    # ── 5. Handler sees prior node outputs ───────────────────────────────

    def test_05_handler_can_read_prior_node_results(
        self, gate_state: dict[str, Any]
    ) -> None:
        """Handler must have access to the DRAFT node's output when it's
        asked to approve the review step."""
        require_env()

        seen_draft: dict[str, Any] = {}

        def review_handler(node, context, state):
            node_results = context.get("node_results") or {}
            seen_draft["draft_node_result"] = node_results.get("draft")
            return True

        chain = _build_chain_draft_review_finalize()
        result = chain.execute(
            {"message": "customer issue"},
            hitl_handler=review_handler,
        )

        draft_seen = seen_draft.get("draft_node_result")
        assert draft_seen is not None, (
            "handler saw no draft result in context['node_results']"
        )
        # The draft tool's result gets wrapped in {output: ..., error: ...}.
        draft_output = (
            draft_seen.get("output") if isinstance(draft_seen, dict) else None
        )
        assert isinstance(draft_output, dict), (
            f"draft_output is not a dict: {draft_output!r}"
        )
        assert "draft_text" in draft_output, (
            f"draft_output missing draft_text: {draft_output}"
        )
        assert (
            draft_output["draft_text"]
            == "Dear customer, your issue will be resolved shortly."
        )
        # And the chain ran to completion past the HITL.
        assert "finalize" in result.node_results, (
            f"finalize node did not run after approval: "
            f"{list(result.node_results.keys())}"
        )

    # ── 6. Approved flow continues / rejection halts ──────────────────────

    def test_06_rejection_halts_chain(
        self, gate_state: dict[str, Any]
    ) -> None:
        """A rejection stops the chain at the gate (1.77.0, audit M9): the
        gate records approved=False, the next node never runs, and the
        result says ``rejected``.

        Until 1.77.0 this test pinned the opposite — the chain kept going —
        and failed loudly when the halt was introduced, as it was meant to.
        If it fails now, rejected work is running again: that is the
        regression, not the test.
        """
        require_env()

        chain = _build_chain_draft_review_finalize()
        result = chain.execute(
            {"message": "customer issue"},
            hitl_handler=lambda n, c, s: False,
        )

        review = result.node_results.get("review") or {}
        assert review.get("approved") is False, (
            f"handler returned False but result shows {review}"
        )
        assert result.status == "rejected", result
        assert "finalize" not in result.node_results, (
            "a rejected approval gate let the next node run"
        )
