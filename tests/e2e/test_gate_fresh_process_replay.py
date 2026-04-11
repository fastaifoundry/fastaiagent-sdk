"""End-to-end quality gate — replay in a fresh Python process.

The main quality gate exercises ``fork_at(2).rerun()`` in the same
process that created the tools, so the ``ToolRegistry`` happens to still
contain them. This gate proves the more realistic scenario: you recorded
a trace in one process, now you want to load it back and rerun it in a
completely different Python process where the original tool callables
were never registered.

Two steps:
1. Parent process: run an agent with a tool, capture the trace_id,
   spawn a subprocess that loads the trace and reruns it WITHOUT
   importing the tool module. The subprocess should exit 0 and report
   a ReplayResult whose rerun surfaces the "tool not registered"
   fallback (the agent receives a "no function attached" error from the
   unregistered tool and can still produce a final answer).

2. Parent process: spawn a second subprocess that DOES register the
   tool before rerunning. That one should succeed with a real answer.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


# The child script is a standalone entry-point we exec in the subprocess.
# It takes a trace_id and an optional "--register" flag to bind the tool.
_CHILD_SCRIPT = r"""
import json
import sys

from fastaiagent.trace.replay import Replay

trace_id = sys.argv[1]
register_tool = "--register" in sys.argv[2:]

if register_tool:
    from fastaiagent import FunctionTool

    def lookup_order(order_id: str) -> str:
        return f"Order {order_id}: shipped on 2026-04-01"

    FunctionTool(name="lookup_order", fn=lookup_order)  # auto-registers

replay = Replay.load(trace_id)
steps_before = len(replay.steps())
forked = replay.fork_at(step=min(2, steps_before - 1))
forked.modify_prompt("Reply in one sentence.")
result = forked.rerun()

print(json.dumps({
    "steps_before": steps_before,
    "new_output": result.new_output,
    "trace_id": result.trace_id,
}))
"""


class TestFreshProcessReplayGate:
    def test_01_record_trace_in_parent(self, gate_state: dict[str, Any]) -> None:
        require_env()
        from fastaiagent import Agent, FunctionTool, LLMClient

        def lookup_order(order_id: str) -> str:
            """Look up an order by ID."""
            return f"Order {order_id}: shipped on 2026-04-01"

        agent = Agent(
            name="fresh-replay-parent",
            system_prompt=(
                "You are a support bot. Use lookup_order when asked about an order."
            ),
            llm=LLMClient(provider="openai", model="gpt-4.1"),
            tools=[FunctionTool(name="lookup_order", fn=lookup_order)],
        )
        result = agent.run("What is the status of order ORD-900?")
        assert result.trace_id, "parent run produced no trace_id"
        assert result.tool_calls, "parent run did not invoke lookup_order"
        gate_state["fresh_trace_id"] = result.trace_id

    def test_02_rerun_in_fresh_subprocess_without_tool_registered(
        self, gate_state: dict[str, Any]
    ) -> None:
        """Fresh process without tool registered: fallback should fire, not crash."""
        require_env()
        trace_id = gate_state["fresh_trace_id"]

        proc = subprocess.run(
            [sys.executable, "-c", _CHILD_SCRIPT, trace_id],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, (
            f"subprocess exited non-zero:\nstdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        assert payload["steps_before"] >= 3, (
            f"child process saw fewer than 3 steps: {payload}"
        )
        # new_output may contain the "tool not registered" fallback string
        # surfaced through the agent, or a hallucinated answer — either
        # way it must be non-empty and the process must not have crashed.
        assert payload["new_output"] is not None, (
            "fresh-process rerun returned new_output=None — "
            "Phase B fallback is broken"
        )
        assert len(payload["new_output"]) > 0

    def test_03_rerun_in_fresh_subprocess_with_tool_registered(
        self, gate_state: dict[str, Any]
    ) -> None:
        """Fresh process with tool re-registered: real rerun succeeds."""
        require_env()
        trace_id = gate_state["fresh_trace_id"]

        proc = subprocess.run(
            [sys.executable, "-c", _CHILD_SCRIPT, trace_id, "--register"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, (
            f"subprocess with --register exited non-zero:\nstdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        assert payload["new_output"], "re-registered rerun produced empty output"
        assert payload["trace_id"], "re-registered rerun emitted no new trace_id"
