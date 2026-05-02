"""Example 48 — Export a trace as a self-contained JSON file.

Runs a small Agent, then writes its trace to ``trace.json`` using the
single-sourced ``build_export_payload`` helper. The same JSON shape
is what ``fastaiagent export-trace`` and the Local UI's Export button
produce.

Usage::

    pip install 'fastaiagent[ui,openai]'
    zsh -lc 'export OPENAI_API_KEY=sk-...'
    zsh -lc 'python examples/48_export_trace.py'

After running, inspect ``trace.json`` — it carries trace metadata, every
span (input / output / attributes / events / model / tokens / cost),
checkpoints (when the trace was durable), and multimodal attachment
metadata. See ``docs/ui/export-trace.md`` for the schema reference and
the screenshot at
``docs/ui/screenshots/sprint1-5-export-dialog.png``.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastaiagent import Agent, LLMClient
from fastaiagent.trace.trace_export import export_trace_to_file


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for this example.")

    agent = Agent(
        name="weather-bot",
        system_prompt="Answer in one short sentence.",
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    )
    result = agent.run("What's a fun fact about octopuses?")
    print("output:", result.output[:200])
    print("trace_id:", result.trace_id)

    db_path = Path(".fastaiagent") / "local.db"
    if not db_path.exists():
        raise SystemExit(
            f"Expected the SDK to have written {db_path}; was tracing disabled?"
        )

    out = export_trace_to_file(
        db_path,
        result.trace_id,
        Path("trace.json"),
    )
    print(f"wrote {out}")
    print("Open it in any JSON viewer or paste it into a GitHub issue.")


if __name__ == "__main__":
    main()
