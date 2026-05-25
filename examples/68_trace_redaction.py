"""Example 68: Opt-in trace redaction (capture + read modes).

The full workflow for masking sensitive values in stored spans:

1. Install a ``RedactionPolicy`` with regex patterns.
2. Run an agent — the LLM response containing a "secret" is captured.
3. Inspect what landed in SQLite — the secret is masked at the storage
   boundary (capture mode), so any downstream OTel exporter would also
   see redacted attributes.
4. Switch to ``mode="read"`` and confirm raw storage but masked reads.

No real LLM call is needed — we use a deterministic ``LLMClient``
subclass so the example runs offline (and exercises the exact same
capture pipeline as a real run would).

Usage::

    python examples/68_trace_redaction.py
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from fastaiagent import Agent
from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.trace import (
    RedactionPolicy,
    TraceStore,
    set_redaction_policy,
)


class _CannedLLM(LLMClient):
    """LLMClient that returns a fixed response (real subclass, not a mock)."""

    def __init__(self, response_text: str) -> None:
        super().__init__(provider="openai", model="gpt-4o-mini", api_key="not-used")
        self._text = response_text

    def _get_provider_fn(self):  # type: ignore[override]
        async def _respond(_messages, _tools=None, **_kwargs):
            return LLMResponse(
                content=self._text,
                finish_reason="stop",
                usage={"input_tokens": 5, "output_tokens": 12},
            )

        return _respond


def _show_stored_attrs(trace_id: str) -> dict:
    """Read back the root agent span from local storage and return its
    attributes dict so we can show whether the secret was stored raw
    or redacted."""
    store = TraceStore.default()
    trace = store.get_trace(trace_id)
    for span in trace.spans:
        if span.name.startswith("agent."):
            return dict(span.attributes)
    return {}


if __name__ == "__main__":
    # Make the local trace.db scoped to this example (so we don't pollute
    # the user's main ~/.fastaiagent/local.db). ``.fastaiagent-demo/`` is
    # already gitignored.
    demo_root = Path(".fastaiagent-demo/redaction")
    demo_root.mkdir(parents=True, exist_ok=True)
    os.environ["FASTAIAGENT_HOME"] = str(demo_root.resolve())

    secret_response = (
        "Here is your one-time API key: sk-DEMO12345678901234567890ABCDEFGH. "
        "Treat it as a credential."
    )

    # ── Step 1: install a capture-mode redaction policy ─────────────────
    print("Step 1: Installing RedactionPolicy(mode='capture')...")
    set_redaction_policy(
        RedactionPolicy(
            patterns=(r"sk-[A-Za-z0-9]{20,}",),
            replacement="[REDACTED]",
            mode="capture",
        )
    )

    # ── Step 2: run a simple agent that leaks a secret ──────────────────
    print("Step 2: Running an agent whose LLM response contains a fake key...")
    agent = Agent(
        name="leaky-bot",
        system_prompt="Echo the user's message.",
        llm=_CannedLLM(secret_response),
    )
    result = agent.run("Give me a key.")

    # OTel exporters batch — wait a tick before reading from SQLite.
    time.sleep(0.1)

    # ── Step 3: confirm storage is redacted ─────────────────────────────
    print(f"Step 3: Reading back trace {result.trace_id} from local storage...")
    attrs_capture = _show_stored_attrs(result.trace_id)
    raw_output = attrs_capture.get("agent.output", "")
    print(f"  agent.output: {raw_output[:80]}...")
    assert "sk-DEMO" not in str(raw_output), (
        "expected the secret to be masked at the storage boundary"
    )
    print("  [REDACTED] confirmed at the storage boundary — downstream")
    print("  OTel exporters added via add_exporter() would also see the")
    print("  masked version.")
    print()

    # ── Step 4: switch to read-mode and rerun ───────────────────────────
    # read-mode keeps storage raw and only masks on the way out (e.g.
    # when the UI is called with ?redact=true). Useful for screen-shares
    # without rewriting on-disk history.
    print("Step 4: Switching to mode='read' for a second run...")
    set_redaction_policy(
        RedactionPolicy(
            patterns=(r"sk-[A-Za-z0-9]{20,}",),
            replacement="[REDACTED]",
            mode="read",
        )
    )
    second = agent.run(f"Run {uuid.uuid4().hex[:6]}: give me another key.")
    time.sleep(0.1)
    attrs_read = _show_stored_attrs(second.trace_id)
    stored_output = attrs_read.get("agent.output", "")
    # In read mode the *stored* value is raw — masking only kicks in at
    # the UI / API boundary when ``?redact=true`` is passed (see
    # tests/test_ui_traces_redaction.py for the HTTP round-trip).
    print(f"  Stored (raw): {stored_output[:80]}...")
    assert "sk-DEMO" in str(stored_output), "read-mode policies must leave storage untouched"
    print()

    # ── Step 5: disable the policy ──────────────────────────────────────
    print("Step 5: set_redaction_policy(None) — back to default (no redaction).")
    set_redaction_policy(None)
    print()

    print("Done. Try setting the policy in your own app — defaults to OFF.")
    print("For the security model overview, see docs/security.md.")
