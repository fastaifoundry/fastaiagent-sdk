"""Example 51 — Guardrail Event Detail demo.

Runs four **real** guardrails inside two real traces so the Local UI's
Guardrails list and its event-detail pages have something to render:

    * ``blocked``  — ``no_pii()`` finds an email address in the output.
    * ``filtered`` — a ``regex`` rule with ``action="mask"`` rewrites the
      payload instead of halting; the detail page diffs before / after.
    * ``warned``   — a rule with ``action="warn"`` fails without stopping
      the run.
    * ``blocked *`` — a ``mask`` the runtime could not carry out. ``code``
      is not a maskable type (it returns a verdict, not offsets), so the
      mask degrades to a block and the list page marks the divergence
      between ``action`` and ``action_taken`` with a ``*``.

No API key is needed: every rule here is a regex / code check.

Why the rules are executed rather than hand-written. Until 1.68.0 this
example INSERTed its rows directly, with a column list that predated the v18
schema — so ``action``, ``action_taken``, ``severity`` and ``floor`` were
always NULL. The Action and Severity columns rendered blank, and the ``*``
marker documented in ``docs/ui/guardrail-events.md`` could never appear on
the very row the example exists to show. Letting the runtime's own writer
(``ui.events.log_guardrail_event``, reached through ``Guardrail.execute``)
produce the rows means the demo cannot drift from the runtime shape again.

Prereqs:
    pip install 'fastaiagent[ui]'

Run:
    python examples/51_guardrail_events.py
    fastaiagent ui --no-auth
    # Open http://127.0.0.1:7842/guardrails
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

# Guardrail events are written only when the Local UI event log is enabled
# (``SDKConfig.ui_enabled``) — users who never open the UI pay nothing for it.
# This demo exists to populate that UI, so switch it on before anything reads
# the config. ``fastaiagent ui`` sets the same flag for its own process.
os.environ.setdefault("FASTAIAGENT_UI_ENABLED", "1")

from fastaiagent._internal.config import get_config, reset_config  # noqa: E402
from fastaiagent.guardrail import no_pii  # noqa: E402
from fastaiagent.guardrail.guardrail import (  # noqa: E402
    Guardrail,
    GuardrailPosition,
    GuardrailResult,
    GuardrailType,
)
from fastaiagent.trace.otel import get_tracer  # noqa: E402

EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"

#: What each ``action`` looks like when it *was* carried out. Mirrors the
#: Local UI's own ``ACTION_FULFILLED`` map (GuardrailsPage.tsx); anything else
#: in ``action_taken`` means the action degraded, which the UI marks with ``*``.
_FULFILLED = {
    "block": "blocked",
    "warn": "warned",
    "mask": "masked",
    "override": "overridden",
    "reask": "reask",
}

BALANCE_REPLY = "Your balance is $42.10. Email me at alice@example.com for follow-up."
FRUSTRATED_INPUT = "I'm really frustrated, this is the third time I've had to call."


def _tone_check(text: str) -> GuardrailResult:
    """A keyword frustration check — the zero-dependency shape of a classifier.

    Scores below the threshold, so the rule fails without the content being
    dangerous. Paired with ``action="warn"`` that is exactly the case the
    ``warned`` outcome exists for: record it, don't halt the run.
    """
    cues = [w for w in ("frustrated", "angry", "furious", "useless") if w in text.lower()]
    if cues:
        return GuardrailResult(
            passed=False,
            score=round(0.16 * len(cues), 2),
            message="Below threshold — passed with note",
            metadata={"cues": cues, "threshold": 0.5},
        )
    return GuardrailResult(passed=True, score=0.0)


def _max_event_rowid(db_path: Path) -> int:
    """Highest ``guardrail_events.rowid`` already stored, or 0 if there is none."""
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM guardrail_events").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def _events_since(db_path: Path, watermark: int) -> list[tuple[str, ...]]:
    """The rows this run wrote — not "the most recent rows", which on a
    lived-in ``local.db`` would be somebody else's."""
    conn = sqlite3.connect(str(db_path))
    try:
        return list(
            conn.execute(
                "SELECT event_id, guardrail_name, outcome, action, action_taken, "
                "COALESCE(severity, '—'), floor "
                "FROM guardrail_events WHERE rowid > ? ORDER BY rowid",
                (watermark,),
            )
        )
    finally:
        conn.close()


def main() -> int:
    reset_config()  # pick up FASTAIAGENT_UI_ENABLED set above
    db_path = Path(get_config().local_db_path)
    watermark = _max_event_rowid(db_path)

    tracer = get_tracer()

    # ── Trace 1 — three rules on one agent.support-bot run. ───────────────
    # The rules run *inside* the span, so ``log_guardrail_event`` picks the
    # trace_id / span_id off the active OTel context exactly as it does in a
    # real agent run. That is what binds each event to the content panel and
    # the execution-context timeline on the detail page.
    with tracer.start_as_current_span("agent.support-bot") as root:
        root.set_attribute("agent.name", "support-bot")
        root.set_attribute("agent.input", "What's my balance?")
        root.set_attribute("agent.output", BALANCE_REPLY)

        # A child LLM span, so the detail page's execution-context timeline
        # has more than one entry to draw.
        with tracer.start_as_current_span("llm.openai.gpt-4o-mini") as llm:
            llm.set_attribute("gen_ai.request.model", "gpt-4o-mini")
            llm.set_attribute("gen_ai.request.messages", "[user] What's my balance?")
            llm.set_attribute("gen_ai.response.content", BALANCE_REPLY)

        # blocked — the stock builtin. Note its type is ``code``, not
        # ``regex``: ``no_pii`` carries its logic in ``fn=`` and delegates to
        # ``safety_detectors.detect_pii``.
        no_pii(position=GuardrailPosition.output).execute(BALANCE_REPLY)

        # filtered — a maskable type, so the rewrite actually happens and the
        # before / after diff is filled in for you.
        Guardrail(
            name="email_redactor",
            guardrail_type=GuardrailType.regex,
            position=GuardrailPosition.output,
            config={"pattern": EMAIL_PATTERN, "should_match": False},
            action="mask",
            severity="medium",
            description="Redacts email addresses from the reply",
        ).execute(BALANCE_REPLY)

        # blocked, with the ``*`` divergence marker — same detection, but a
        # ``code`` rule cannot tell the runtime *where* the match was, so the
        # configured mask degrades to a block. A safety behaviour, not a
        # fault; the list page flags it so an over-blocking rule is visible.
        Guardrail(
            name="no_pii_strict",
            guardrail_type=GuardrailType.code,
            position=GuardrailPosition.output,
            fn=no_pii().fn,
            action="mask",
            severity="critical",
            floor=True,
            description="Organisation baseline PII rule",
        ).execute(BALANCE_REPLY)

    # ── Trace 2 — a warn on a different run / different rule. ─────────────
    with tracer.start_as_current_span("agent.support-bot") as root2:
        root2.set_attribute("agent.name", "support-bot")
        root2.set_attribute("agent.input", FRUSTRATED_INPUT)
        root2.set_attribute(
            "agent.output",
            "I hear you — let me help. Could you tell me what went wrong?",
        )
        Guardrail(
            name="tone_watch",
            guardrail_type=GuardrailType.code,
            position=GuardrailPosition.input,
            fn=_tone_check,
            action="warn",
            severity="low",
            description="Flags frustration cues without halting the run",
        ).execute(FRUSTRATED_INPUT)

    rows = _events_since(db_path, watermark)
    if not rows:
        print(f"No guardrail events were written to {db_path}.")
        print("Check that FASTAIAGENT_UI_ENABLED is not set to 0 in your environment.")
        return 1

    print(f"Wrote {len(rows)} guardrail events to {db_path}\n")
    print(f"  {'rule':<16}{'outcome':<11}{'action':<9}{'taken':<12}{'severity':<10}floor")
    for _eid, name, outcome, action, taken, severity, floor in rows:
        star = " *" if action and taken and _FULFILLED.get(action) != taken else ""
        print(
            f"  {name:<16}{outcome:<11}{action:<9}{(taken + star):<12}"
            f"{severity:<10}{'yes' if floor else 'no'}"
        )

    print("\nOpen the Local UI:")
    print("  fastaiagent ui --no-auth")
    print("\nDirect links once it's running on port 7842:")
    print("  http://127.0.0.1:7842/guardrails")
    for eid, name, *_ in rows:
        print(f"  http://127.0.0.1:7842/guardrail-events/{eid}   ({name})")
    print(
        "\nTip: open the blocked event, click 'Mark as false positive',"
        " refresh — the flag persists."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
