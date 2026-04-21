"""Example 41 — Agents × Tools: what's registered, what's actually used.

Produces a trace with a mixed toolkit so the /agents/<name>/tools view
has something to show:

  - a ``@tool`` decorator function           → origin "function"
  - ``LocalKB.as_tool()``                    → origin "kb"
  - a user-defined ``Tool`` subclass         → origin "custom"

Then runs the agent twice with different queries so **one** of the
registered tools is never called — the UI flags it as ``unused``.

Prereqs:
    pip install 'fastaiagent[ui,openai,kb]'
    export OPENAI_API_KEY=sk-...

Run:
    python examples/41_agent_tools.py
    fastaiagent ui   # /agents/tool-curator → Tools section
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastaiagent import Agent, LLMClient, tool
from fastaiagent.kb.local import LocalKB
from fastaiagent.tool.base import Tool, ToolResult

# ─── A) @tool decorator ───────────────────────────────────────────────────


@tool(description="Convert a temperature in Celsius to Fahrenheit.")
def c_to_f(celsius: float) -> float:
    return celsius * 9 / 5 + 32


# ─── B) Custom Tool subclass ──────────────────────────────────────────────


class PunctuationCounter(Tool):
    """A bespoke Tool subclass counting punctuation in a string.

    Shows up with the 'custom' origin chip in the UI because we don't
    override ``Tool.origin`` (the class attribute defaults to "custom").
    """

    def __init__(self) -> None:
        super().__init__(
            name="count_punctuation",
            description="Count punctuation characters in a text.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )

    async def aexecute(self, arguments: dict, context=None) -> ToolResult:  # type: ignore[override]
        text = arguments.get("text", "")
        marks = sum(1 for c in text if c in ".,;:!?\"'")
        return ToolResult(output={"count": marks})


# ─── C) LocalKB.as_tool() ─────────────────────────────────────────────────


def _seed_kb() -> LocalKB:
    source_dir = Path(".fastaiagent-source-docs")
    source_dir.mkdir(exist_ok=True)
    (source_dir / "refunds.md").write_text(
        "# Refund policy\n\nRefunds are processed within 7 business days.\n"
    )
    kb = LocalKB(name="tool-demo-kb", chunk_size=240, chunk_overlap=30)
    kb.add(str(source_dir / "refunds.md"))
    return kb


# ─── Run ──────────────────────────────────────────────────────────────────


def require_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required for this example.")
        sys.exit(0)


def main() -> None:
    require_key()
    kb = _seed_kb()

    agent = Agent(
        name="tool-curator",
        system_prompt=(
            "You are a helpful assistant with a mixed toolkit. Use any tool "
            "that applies. Be concise."
        ),
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        tools=[c_to_f, PunctuationCounter(), kb.as_tool()],
    )

    print("► query 1 — should call the @tool function")
    r1 = agent.run("What is 20 °C in Fahrenheit? Use the tool.")
    print(f"  output: {str(r1.output)[:100]}")

    print("► query 2 — should call the KB tool")
    r2 = agent.run("When do refunds get processed? Check the knowledge base.")
    print(f"  output: {str(r2.output)[:100]}")

    # count_punctuation is intentionally never asked for — the Tools view
    # in the UI will show it with an 'unused' badge.

    base = os.environ.get("FASTAIAGENT_UI_URL", "http://127.0.0.1:7842")
    print()
    print("Done. Open the UI:")
    print(f"  {base}/agents/tool-curator")
    print()
    print("In the Tools section you'll see three rows:")
    print("  • c_to_f               [function chip] — 1 call")
    print("  • search_tool-demo-kb  [kb chip]       — 1 call")
    print("  • count_punctuation    [custom chip]   — 0 calls → 'unused' badge")


if __name__ == "__main__":
    main()
