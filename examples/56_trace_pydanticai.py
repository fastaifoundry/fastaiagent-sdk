"""Example 56 — Auto-trace a PydanticAI agent.

PydanticAI ships its own GenAI-semconv OpenTelemetry instrumentation
(``Agent.instrument_all()``); ``pa.enable()`` flips that on AND tags
every root span with ``fastaiagent.framework=pydanticai`` so the Local
UI's framework filter can find it. Switches model to Anthropic when
``ANTHROPIC_API_KEY`` is set, otherwise falls back to OpenAI — proves
provider neutrality of the harness.

Run:
    pip install "fastaiagent[pydanticai,ui]"
    OPENAI_API_KEY=sk-...   python examples/56_trace_pydanticai.py
    # or
    ANTHROPIC_API_KEY=sk-... python examples/56_trace_pydanticai.py
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    if not (has_anthropic or has_openai):
        print(
            "Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set — skipping example."
        )
        return 0

    try:
        from pydantic_ai import Agent
    except ImportError:
        print(
            'pydantic-ai is not installed. Install with: '
            'pip install "fastaiagent[pydanticai]"'
        )
        return 0

    from fastaiagent.integrations import pydanticai as pa
    from fastaiagent.trace.storage import TraceStore

    pa.enable()

    # Prefer the Anthropic path so the example is also a smoke test of
    # provider neutrality. The OpenAI path is the fallback.
    model = (
        "anthropic:claude-haiku-4-5" if has_anthropic else "openai:gpt-4o-mini"
    )
    agent = Agent(model, system_prompt="Answer in one word.")

    result = agent.run_sync("What colour is the sky on a clear day?")
    print("output:", result.output)

    store = TraceStore.default()
    for summary in store.list_traces()[:1]:
        trace = store.get_trace(summary.trace_id)
        print(f"\nTrace {summary.trace_id} — {len(trace.spans)} spans:")
        for span in trace.spans:
            print(f"  {span.name}")
        print(
            "Open in the Local UI: "
            f"http://127.0.0.1:7842/traces/{summary.trace_id}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
