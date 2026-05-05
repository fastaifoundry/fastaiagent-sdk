"""
Tools — what each Chain node calls.

Three tool nodes carry the pipeline:

  * ``load_transcript``   — read text or PDF off disk into a normalised
    ``{"text": str, "title": str | None, "date": str | None}`` envelope.

  * ``analyze_meeting``   — fan out to three single-purpose LLM agents
    *in parallel* via ``asyncio.gather``. Each agent returns JSON; the
    tool collects + parses + returns one merged dict. This is the
    "parallel branches" pattern the example showcases — done inside a
    single tool node rather than via ``NodeType.parallel`` because the
    chain executor's parallel-node input contract requires the whole
    initial-state dict get stringified for every child agent. Doing it
    in one tool keeps the per-agent input clean (just the transcript).

  * ``draft_followup_email`` — given the merged ``MeetingNotes`` and one
    attendee's name, write a short personalised followup. Mock or real
    SendGrid backend, same pattern as the SDR template.

Plus ``merge_into_notes`` — a sync helper that runs Pydantic validation
on the analyzer's output and falls back to a partial ``MeetingNotes`` if
any field doesn't parse. Used by ``workflow.py`` as a transformer-style
tool node.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fastaiagent as fa

from schema import ActionItem, Decision, MeetingNotes

_HERE = Path(__file__).resolve().parent
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n|\n```\s*$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def _parse_json_loose(text: Any) -> Any:
    if isinstance(text, (dict, list)):
        return text
    try:
        return json.loads(_strip_fences(text))
    except Exception:
        return {}


# ─── Shared deps ─────────────────────────────────────────────────────────────


@dataclass
class MeetingDeps:
    notes: list[str] = field(default_factory=list)


def make_deps() -> MeetingDeps:
    return MeetingDeps()


# ─── Lazy analyzer agents ────────────────────────────────────────────────────


_summarizer: fa.Agent | None = None
_action_extractor: fa.Agent | None = None
_decision_extractor: fa.Agent | None = None


def _build_agents(llm: fa.LLMClient) -> tuple[fa.Agent, fa.Agent, fa.Agent]:
    """Build the three analyzer agents on first use."""
    global _summarizer, _action_extractor, _decision_extractor
    if _summarizer is None:
        from workflow import (  # local import — workflow.py imports tools.py
            ACTION_PROMPT,
            DECISION_PROMPT,
            SUMMARIZER_PROMPT,
        )

        _summarizer = fa.Agent(
            name="meeting-summarizer", system_prompt=SUMMARIZER_PROMPT, llm=llm
        )
        _action_extractor = fa.Agent(
            name="action-extractor", system_prompt=ACTION_PROMPT, llm=llm
        )
        _decision_extractor = fa.Agent(
            name="decision-extractor", system_prompt=DECISION_PROMPT, llm=llm
        )
    assert _summarizer and _action_extractor and _decision_extractor
    return _summarizer, _action_extractor, _decision_extractor


# ─── Transcript loader ───────────────────────────────────────────────────────


@fa.tool()
def load_transcript(path: str, ctx: fa.RunContext[MeetingDeps]) -> dict:
    """Read a meeting transcript from disk.

    Supported: ``.md`` / ``.txt`` (plain text) and ``.pdf`` (extracted via
    ``fa.PDF.extract_text`` — uses pypdf under the hood). Returns a dict
    with keys ``text``, ``title``, ``date`` — title and date are best-effort
    parsed from the first heading and a ``date:`` line if present.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {"error": f"File not found: {path}"}

    if p.suffix.lower() == ".pdf":
        from fastaiagent.multimodal.pdf import PDF

        text = PDF.from_file(p).extract_text()
    else:
        text = p.read_text()

    # Best-effort title + date extraction. Falls through to the LLM-led
    # extraction inside the analyzer agents if the heuristics don't match.
    title = None
    date = None
    first_lines = text.splitlines()[:6]
    for line in first_lines:
        if line.startswith("# ") and title is None:
            heading = line.lstrip("# ").strip()
            # Pull a trailing "— YYYY-MM-DD" off the heading if present
            m = re.match(r"^(.*?)\s+[—-]\s+(\d{4}-\d{2}-\d{2})\s*$", heading)
            if m:
                title, date = m.group(1).strip(), m.group(2)
            else:
                title = heading
        m_date = re.match(r"^(?:date|when):\s*(.+)$", line, re.IGNORECASE)
        if m_date and date is None:
            date = m_date.group(1).strip()

    return {"text": text, "title": title, "date": date, "path": str(p)}


# ─── Parallel analyzer ───────────────────────────────────────────────────────


@fa.tool()
async def analyze_meeting(transcript: str, ctx: fa.RunContext[MeetingDeps]) -> dict:
    """Run the three analyzer agents concurrently against the transcript.

    Returns a dict with the raw + parsed outputs from each agent. The
    merge step (``merge_into_notes``) validates these against the
    ``MeetingNotes`` Pydantic model.
    """
    llm = fa.LLMClient(provider="openai", model=os.getenv("LLM_MODEL", "gpt-4o"))
    summarizer, action_extractor, decision_extractor = _build_agents(llm)

    # Three concurrent LLM calls. asyncio.gather collects errors as
    # exceptions in the result list; we bubble each agent's failure as
    # an empty payload so the merge step still produces a partial
    # MeetingNotes rather than crashing the whole chain.
    summary_task = summarizer.arun(transcript, context=ctx)
    actions_task = action_extractor.arun(transcript, context=ctx)
    decisions_task = decision_extractor.arun(transcript, context=ctx)

    summary_r, actions_r, decisions_r = await asyncio.gather(
        summary_task, actions_task, decisions_task, return_exceptions=True
    )

    def _safe_output(r: object) -> str:
        if isinstance(r, Exception):
            return ""
        return getattr(r, "output", "") or ""

    return {
        "summary_raw": _safe_output(summary_r),
        "actions_raw": _safe_output(actions_r),
        "decisions_raw": _safe_output(decisions_r),
    }


# ─── Merge into structured Pydantic ──────────────────────────────────────────


@fa.tool()
def merge_into_notes(
    transcript_meta: str,
    analysis: str,
    ctx: fa.RunContext[MeetingDeps],
) -> dict:
    """Validate + merge the analyzer outputs into a ``MeetingNotes`` model.

    ``transcript_meta`` is the JSON-stringified output of ``load_transcript``;
    ``analysis`` is the JSON-stringified output of ``analyze_meeting``. Both
    are passed via ``input_mapping`` from the chain. Returns the validated
    ``MeetingNotes`` as a dict.

    On per-field parse failures we fall through to a partial model rather
    than raise — keeps the chain run useful even when one analyzer
    returned malformed JSON.
    """
    meta = _parse_json_loose(transcript_meta)
    raw = _parse_json_loose(analysis)

    # Each *_raw is itself a JSON string from an LLM. Parse each defensively.
    summary_payload = _parse_json_loose(raw.get("summary_raw", ""))
    actions_payload = _parse_json_loose(raw.get("actions_raw", ""))
    decisions_payload = _parse_json_loose(raw.get("decisions_raw", ""))

    summary_text = ""
    attendees: list[str] = []
    if isinstance(summary_payload, dict):
        summary_text = str(summary_payload.get("summary", ""))
        a = summary_payload.get("attendees") or []
        if isinstance(a, list):
            attendees = [str(x) for x in a]

    action_items: list[ActionItem] = []
    if isinstance(actions_payload, dict):
        for entry in actions_payload.get("action_items", []) or []:
            try:
                action_items.append(ActionItem.model_validate(entry))
            except Exception:
                continue

    decisions: list[Decision] = []
    if isinstance(decisions_payload, dict):
        for entry in decisions_payload.get("decisions", []) or []:
            try:
                decisions.append(Decision.model_validate(entry))
            except Exception:
                continue

    notes = MeetingNotes(
        title=str(meta.get("title") or "Untitled Meeting"),
        date=meta.get("date"),
        attendees=attendees,
        summary=summary_text,
        action_items=action_items,
        decisions=decisions,
    )
    return notes.model_dump()


# ─── Followup email tools ────────────────────────────────────────────────────


def _email_send_mock(*, to: str, subject: str, body: str) -> dict:
    log_path = _HERE / ".fastaiagent" / "outbox.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    msg_id = f"MSG-{uuid.uuid4().hex[:10]}"
    with log_path.open("a") as f:
        f.write(json.dumps({"msg_id": msg_id, "to": to, "subject": subject, "body": body}) + "\n")
    return {"sent": True, "msg_id": msg_id, "to": to, "ts": time.time()}


def _email_send_sendgrid(*, to: str, subject: str, body: str) -> dict:
    import httpx

    api_key = os.getenv("SENDGRID_API_KEY")
    sender = os.getenv("SENDGRID_FROM")
    if not (api_key and sender):
        raise RuntimeError("EMAIL_BACKEND=sendgrid but credentials missing.")
    response = httpx.post(
        "https://api.sendgrid.com/v3/mail/send",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "personalizations": [{"to": [{"email": to}]}],
            "from": {"email": sender},
            "subject": subject,
            "content": [{"type": "text/plain", "value": body}],
        },
        timeout=20.0,
    )
    response.raise_for_status()
    return {"sent": True, "msg_id": response.headers.get("X-Message-Id", ""), "to": to}


_EMAIL_BACKENDS = {"mock": _email_send_mock, "sendgrid": _email_send_sendgrid}


@fa.tool()
async def draft_followup_email(
    notes_json: str,
    attendee_name: str,
    ctx: fa.RunContext[MeetingDeps],
) -> dict:
    """Compose + send a per-attendee followup email.

    Calls a small drafter agent with the slice of the notes that's
    relevant to ``attendee_name`` (their action items + the decisions
    that affected them). Returns a send receipt; the chain does NOT
    pause for HITL here — followup emails go straight out via the
    configured ``EMAIL_BACKEND``. Add an ``fa.interrupt()`` if your
    org's policy requires approval before any meeting-followup goes out.
    """
    from workflow import FOLLOWUP_PROMPT

    notes = MeetingNotes.model_validate(_parse_json_loose(notes_json))
    slice_dict = notes.for_attendee(attendee_name)
    if not slice_dict["action_items"]:
        return {"sent": False, "skipped": True, "reason": "no action items for this attendee"}

    llm = fa.LLMClient(provider="openai", model=os.getenv("LLM_MODEL", "gpt-4o"))
    drafter = fa.Agent(name="followup-drafter", system_prompt=FOLLOWUP_PROMPT, llm=llm)

    result = await drafter.arun(json.dumps(slice_dict), context=ctx)
    parsed = _parse_json_loose(result.output)
    subject = str(parsed.get("subject", f"Followup: {notes.title}"))
    body = str(parsed.get("body", result.output))

    # Dispatch via configured backend.
    to = parsed.get("to") or f"{attendee_name.lower().replace(' ', '.')}@example.com"
    backend = os.getenv("EMAIL_BACKEND", "mock")
    fn = _EMAIL_BACKENDS.get(backend, _email_send_mock)
    return fn(to=to, subject=subject, body=body)
