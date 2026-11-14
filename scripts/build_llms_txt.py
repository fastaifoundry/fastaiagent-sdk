#!/usr/bin/env python3
"""Write ``docs/llms.txt`` and ``docs/llms-full.txt`` from the packaged skill.

    python scripts/build_llms_txt.py           # rewrite both files
    python scripts/build_llms_txt.py --check   # exit 1 if either is out of date

``llms.txt`` is a short index in the llmstxt.org shape; ``llms-full.txt`` is the skill itself
(``SKILL.md`` then every reference, in order) as one page, so the two never disagree with what
``fastaiagent skills install`` ships. No tool reads these on its own — they are an index the
skill and the docs link to. mkdocs copies both to the site root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILL = REPO / "fastaiagent" / "skills" / "fastaiagent-sdk"
SITE = "https://docs.fastaiagent.net"

# (title, docs path, one line)
PAGES = [
    ("Getting started", "getting-started/", "Install, first agent, coding-assistant skills"),
    ("Agents", "agents/", "Agent, LLMClient, AgentConfig, middleware, memory, teams, swarm"),
    ("Tools", "tools/", "@tool, REST tools, MCP tools, replay_class"),
    ("Chains", "chains/", "Graph workflows: cycles, typed nodes, human approval"),
    ("Guardrails", "guardrails/", "Input, output and tool-call safety checks"),
    ("Evaluation", "evaluation/", "evaluate(), scorers, the pytest plugin, CI gates"),
    ("Testing without a model", "testing/", "TestModel and FunctionModel"),
    ("Tracing and replay", "tracing/", "Local traces, OpenTelemetry, replay"),
    ("Platform", "platform/", "connect(), prompts, push, the runner"),
    ("CLI", "cli/", "Every fastaiagent command"),
]


def _split(text: str) -> tuple[str, str]:
    head, _, body = text[4:].partition("\n---\n")
    return head, body.lstrip("\n")


def _reference_order() -> list[Path]:
    """References in the order SKILL.md's map lists them, then any not listed."""
    skill_md = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    refs = {p.name: p for p in sorted((SKILL / "references").glob("*.md"))}
    ordered: list[Path] = []
    for line in skill_md.splitlines():
        for name in list(refs):
            if f"references/{name}" in line:
                ordered.append(refs.pop(name))
    return ordered + list(refs.values())


def build_full() -> str:
    skill_md = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    head, body = _split(skill_md)
    description = next(
        (ln.split(":", 1)[1].strip() for ln in head.splitlines() if ln.startswith("description:")),
        "",
    )
    parts = [
        "# FastAIAgent SDK — full reference for coding assistants\n",
        f"> {description}\n",
        "This page is the `fastaiagent-sdk` skill: the main page, then one reference per area. "
        "Install it into a project with `fastaiagent skills install`.\n",
        "---\n",
        body.rstrip() + "\n",
    ]
    for ref in _reference_order():
        parts += ["---\n", ref.read_text(encoding="utf-8").rstrip() + "\n"]
    return "\n".join(parts)


def build_index() -> str:
    lines = [
        "# FastAIAgent SDK",
        "",
        "> Python SDK to build, debug, evaluate and operate AI agents. Apache-2.0. "
        "Runs standalone; connects to a FastAIAgent platform optionally.",
        "",
        "Assistants: read the full reference first. It lists the mistakes assistants make "
        "with this SDK, then covers each area.",
        "",
        "## Read first",
        "",
        f"- [Full reference for coding assistants]({SITE}/llms-full.txt): "
        "the ten rules, then agents, tools, teams, guardrails, evals, platform, testing, "
        "what changed",
        f"- [Use fastaiagent with a coding assistant]({SITE}/getting-started/coding-assistants/): "
        "install the skill (`fastaiagent skills install`)",
        "",
        "## Documentation",
        "",
    ]
    lines += [f"- [{title}]({SITE}/{path}): {desc}" for title, path, desc in PAGES]
    lines += [
        "",
        "## Source",
        "",
        "- [fastaiagent-sdk](https://github.com/fastaifoundry/fastaiagent-sdk)",
        "- [Agent Skills for coding assistants](https://github.com/fastaifoundry/fastaiagent-skills)",
        "",
    ]
    return "\n".join(lines)


def outputs() -> dict[Path, str]:
    return {
        REPO / "docs" / "llms.txt": build_index(),
        REPO / "docs" / "llms-full.txt": build_full(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if a file is out of date")
    args = ap.parse_args(argv)
    stale = []
    for path, text in outputs().items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current != text:
            stale.append(path)
            if not args.check:
                path.write_text(text, encoding="utf-8")
    for p in stale:
        print(f"{'out of date' if args.check else 'wrote'}: {p.relative_to(REPO)}")
    return 1 if (args.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
