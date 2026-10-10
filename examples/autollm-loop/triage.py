"""The ticket-triage agent, its scorer and its data — shared by every step.

The agent is always built **from the prompt registry**: the registry version is
what runs in production, what each trace is stamped with, and what AutoLLM's
winner becomes. Run every step from this folder, so they all share
``./.fastaiagent/local.db`` — the same store ``fastaiagent ui`` reads.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import fastaiagent as fa
from fastaiagent._internal.config import get_config
from fastaiagent.eval.scorer import Scorer, ScorerResult
from fastaiagent.optimize import Candidate, apply_candidate
from fastaiagent.prompt import Prompt, PromptRegistry

HERE = Path(__file__).resolve().parent
TICKETS = HERE / "data" / "tickets.jsonl"
OUT = HERE / "out"  # hand-off files between steps (gitignored)

PROMPT = "ticket-triage"  # the registry prompt
AGENT = "ticket-triage"  # the agent name traces are filed under
DATASET = "ticket-triage"  # the dataset file the UI's Dataset Editor shows

# The agent runs on a small, cheap model. The prompts are written by a reasoning
# model — working out a house rule from labelled failures is induction, and that
# is what reasoning models are good at.
MODEL = os.environ.get("TRIAGE_MODEL", "gpt-4.1-mini")
PROPOSER_MODEL = os.environ.get("TRIAGE_PROPOSER_MODEL", "gpt-5")

# v1 — what a team would write on day one: the queues, the scale, the format.
# None of the house rules (refund thresholds, plan-based outage priority, ...)
# are in it. They live only in how the support leads label tickets.
V1_PROMPT = """You are the triage agent for Acme Cloud support.
Route each customer ticket to exactly one queue:
- billing: invoices, charges, refunds
- technical: outages, errors, API and product questions
- account: sign-in, access and account settings
- shipping: orders of hardware security keys
- security: suspicious activity and compromised credentials
Set a priority: P1 (urgent), P2 (soon) or P3 (normal).
Reply with JSON only: {"queue": "<queue>", "priority": "<P1|P2|P3>"}"""


def load_agent(*, alias: str | None = "production", version: int | None = None) -> fa.Agent:
    """Build the agent from a registry version (``version``) or alias (``alias``).

    The ``Prompt`` itself is passed — not its text — so every model call is
    stamped with ``fastaiagent.prompt.name`` / ``.version`` and shows up in the
    prompt's lineage on the UI's Prompts page. A version AutoLLM produced also
    carries the few-shot examples it selected (``metadata["fewshot_demos"]``);
    they are rebuilt here, so a registry version is the *whole* winner.
    """
    reg = PromptRegistry()
    prompt = reg.load(PROMPT, version=version) if version else reg.load(PROMPT, alias=alias)
    agent = fa.Agent(
        name=AGENT,
        system_prompt=prompt,
        llm=fa.LLMClient(provider="openai", model=MODEL),
    )
    demos = prompt.metadata.get("fewshot_demos")
    if demos:
        # The call report.apply_to() makes: the examples go in, and the agent
        # stays as stateless as it was — no conversation carried between tickets.
        agent = apply_candidate(agent, Candidate(fewshot_demos=demos))
    return agent


def current_version(alias: str = "production") -> Prompt:
    return PromptRegistry().load(PROMPT, alias=alias)


def tickets() -> list[dict[str, Any]]:
    """The 120 tickets: ``{"id", "ticket", "label": {"queue", "priority"}}``."""
    return [json.loads(line) for line in TICKETS.read_text().splitlines() if line.strip()]


def label_text(label: dict[str, str]) -> str:
    """The exact reply a correct agent gives — the dataset's ``expected_output``."""
    return json.dumps({"queue": label["queue"], "priority": label["priority"]})


def datasets_dir() -> Path:
    """Where the UI's Dataset Editor reads and writes (mirrors the UI's rule)."""
    db = Path(get_config().local_db_path)
    if db.parent.name == ".fastaiagent":
        return db.parent / "datasets"
    return Path.cwd() / ".fastaiagent" / "datasets"


def dataset_path() -> Path:
    return datasets_dir() / f"{DATASET}.jsonl"


def save_json(name: str, data: Any) -> Path:
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(data, indent=2))
    return path


def load_json(name: str) -> Any:
    return json.loads((OUT / name).read_text())


def _parse(text: str | None) -> dict[str, Any] | None:
    try:
        value = json.loads(text or "")
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


class TriageMatch(Scorer):
    """Both fields right = pass. Half credit for one; the reason names the miss.

    Strict JSON on purpose: the reply feeds a ticketing system, and a reply that
    doesn't parse routes nowhere. The reason (``priority P2, expected P1``) is
    what AutoLLM's proposer reads next to the expected label — enough for it to
    work out the house rules the labels encode.
    """

    name = "triage_match"

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        want = _parse(expected)
        if want is None:
            return ScorerResult(score=0.0, passed=False, reason="no expected label")
        got = _parse(output)
        if got is None:
            return ScorerResult(
                score=0.0, passed=False, reason=f"not a JSON object: {str(output)[:60]!r}"
            )
        misses = [
            f"{field} {got.get(field)}, expected {want[field]}"
            for field in ("queue", "priority")
            if got.get(field) != want[field]
        ]
        return ScorerResult(
            score=(2 - len(misses)) / 2,
            passed=not misses,
            reason="; ".join(misses) or None,
        )
