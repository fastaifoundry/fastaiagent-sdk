"""Smoke tests — no live LLM calls.

Fast feedback for anyone iterating on the template before they spend
tokens running ``verify.py``. Covers:

  * top-level module imports
  * the buggy vs fixed tool behaviors against known order IDs
  * agent construction shape (tool wiring, prompt, llm)
  * dataset JSONL has the field names ``evaluate()`` expects
  * ``ForkedReplay.with_tool_override`` swaps the tool by name when
    given a synthetic in-memory trace (real ``LLMClient`` subclass —
    matches the project's no-mocking rule)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add the template root to sys.path so ``import agent`` / ``import tools``
# work whether pytest is run from inside the template dir or the repo root.
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import agent as agent_module  # noqa: E402
import pytest  # noqa: E402
import tools as tools_module  # noqa: E402

from fastaiagent import Agent  # noqa: E402
from fastaiagent.tool import FunctionTool  # noqa: E402

# ── Module imports ──────────────────────────────────────────────────────────


def test_modules_import_cleanly():
    # If any of these break, the whole template breaks — catch early.
    import analyze  # noqa: F401
    import capture  # noqa: F401
    import fix  # noqa: F401
    import save_test  # noqa: F401
    import verify  # noqa: F401


# ── Tool behavior ───────────────────────────────────────────────────────────


class TestBuggyTool:
    def test_known_order_returns_correct_record(self):
        out = tools_module._lookup_order_buggy("ORD-001")
        assert isinstance(out, dict)
        assert out["id"] == "ORD-001"
        assert "MacBook" in out["item"]

    def test_missing_order_returns_fallback_record_stamped_with_requested_id(self):
        # This is the bug — a missing ID silently returns ORD-001's
        # record but with the requested ID stamped on, so the LLM has
        # no way to cross-check. The agent confidently reports wrong
        # details to the user.
        out = tools_module._lookup_order_buggy("ORD-999")
        assert out["id"] == "ORD-999"  # stamped on for plausibility
        assert "MacBook" in out["item"]  # actually ORD-001's data
        assert "error" not in out


class TestFixedTool:
    def test_known_order_returns_structured_dict(self):
        out = tools_module._lookup_order_fixed("ORD-001")
        assert isinstance(out, dict)
        assert out["id"] == "ORD-001"
        assert "MacBook" in out["item"]

    def test_missing_order_returns_error_dict(self):
        out = tools_module._lookup_order_fixed("ORD-999")
        assert out == {"error": "order ORD-999 not found"}


# ── Tool factory shapes ─────────────────────────────────────────────────────


class TestToolFactories:
    def test_buggy_tool_is_named_lookup_order(self):
        t = tools_module.buggy_lookup_order_tool()
        assert isinstance(t, FunctionTool)
        assert t.name == "lookup_order"

    def test_fixed_tool_has_matching_name_so_override_swaps_cleanly(self):
        # ``with_tool_override("lookup_order", fixed)`` only works if the
        # fixed tool's name matches the original. Lock that.
        assert (
            tools_module.fixed_lookup_order_tool().name
            == tools_module.buggy_lookup_order_tool().name
        )


# ── Agent construction (no LLM call — just builder shape) ───────────────────


class TestAgentBuilders:
    def test_buggy_agent_wires_the_buggy_tool(self):
        a = agent_module.build_buggy_agent()
        assert isinstance(a, Agent)
        assert a.name == "support-bot"
        assert len(a.tools) == 1
        assert a.tools[0].name == "lookup_order"

    def test_fixed_agent_wires_the_fixed_tool(self):
        a = agent_module.build_fixed_agent()
        assert a.name == "support-bot"
        assert a.tools[0].name == "lookup_order"

    def test_both_agents_share_prompt_so_only_the_tool_changes(self):
        a = agent_module.build_buggy_agent()
        b = agent_module.build_fixed_agent()
        assert a.system_prompt == b.system_prompt


# ── Dataset shape ───────────────────────────────────────────────────────────


class TestRegressionDataset:
    def test_jsonl_lines_have_evaluate_compatible_keys(self):
        dataset = _HERE / "regression_dataset.jsonl"
        assert dataset.exists(), "Seeded regression_dataset.jsonl missing"
        for raw in dataset.read_text().splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            assert "input" in row
            assert "expected_output" in row

    def test_dataset_includes_the_demo_failure_case(self):
        dataset = _HERE / "regression_dataset.jsonl"
        cases = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
        # The 999 case is the one capture.py / fix.py / save_test.py
        # build up; the dataset ships pre-seeded with it so verify.py
        # works even before the user runs the full loop.
        ord999 = [c for c in cases if "ORD-999" in c["input"]]
        assert ord999, "expected ORD-999 case in seeded dataset"


# ── Replay tool-override integration ────────────────────────────────────────


def test_with_tool_override_swaps_by_name():
    """End-to-end of the API the template depends on, against an
    in-memory ``TraceData`` — no SQLite, no network. Confirms that
    ``ForkedReplay.with_tool_override("lookup_order", fixed_tool)``
    substitutes the tool when the reconstructed agent has one with the
    matching name."""
    from fastaiagent.trace.replay import ForkedReplay
    from fastaiagent.trace.storage import SpanData, TraceData

    trace = TraceData(
        trace_id="t",
        name="x",
        start_time="",
        end_time="",
        spans=[
            SpanData(
                span_id="s",
                trace_id="t",
                name="agent.support-bot",
                start_time="",
                end_time="",
                attributes={},
            )
        ],
    )
    forked = ForkedReplay(original_trace=trace, fork_point=0, steps=[])
    fixed_tool = tools_module.fixed_lookup_order_tool()
    forked.with_tool_override("lookup_order", fixed_tool)

    # Reconstructed agent has the original tool. Override should swap it.
    original_tool = tools_module.buggy_lookup_order_tool()
    out = forked._apply_tool_overrides([original_tool])
    assert out == [fixed_tool]


def test_with_tool_override_requires_non_empty_name():
    from fastaiagent._internal.errors import ReplayError
    from fastaiagent.trace.replay import ForkedReplay
    from fastaiagent.trace.storage import TraceData

    forked = ForkedReplay(
        original_trace=TraceData(trace_id="", name="", start_time="", end_time="", spans=[]),
        fork_point=0,
        steps=[],
    )
    with pytest.raises(ReplayError):
        forked.with_tool_override("", object())
