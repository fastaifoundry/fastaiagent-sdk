"""
Tools — what each Chain node calls.

Three pluggable backends (enrichment, CRM, email), each with a mock that
runs offline and stubs for the canonical real provider. Flip the matching
``*_BACKEND`` env var to wire them up.

The interesting tools here are:

  * ``score_lead`` and ``draft_outreach`` — these wrap LLM agent calls
    *inside* a tool function. The Chain DAG only sees tool nodes; the
    JSON each tool returns is what flows downstream as ``state.output``.

  * ``send_outreach_email`` — calls ``fa.interrupt()`` mid-execution so the
    chain pauses for human approval before the email is actually sent. The
    inner ``_persist_send`` is ``@idempotent`` keyed on (to, subject, body)
    so a resume after approval (or a Replay rerun) reuses the same msg_id
    rather than firing twice.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import fastaiagent as fa
from fastaiagent.chain.idempotent import idempotent

_HERE = Path(__file__).resolve().parent

# ─── Mock data corpus (lead enrichment) ──────────────────────────────────────


_MOCK_LEADS: dict[str, dict] = {
    "alice@acme-saas.com": {
        "name": "Alice Chen",
        "title": "VP of Engineering",
        "company": "Acme SaaS",
        "industry": "B2B SaaS",
        "company_size": "250 employees",
        "funding_stage": "Series B",
        "stack": ["Python", "TypeScript", "OpenAI", "Datadog"],
        "linkedin": "https://linkedin.com/in/alice-chen-acme",
        "country": "United States",
    },
    "bob@indiehacker.dev": {
        "name": "Bob Solo",
        "title": "Indie Hacker",
        "company": "Solo Studio",
        "industry": "Consulting",
        "company_size": "1 employee",
        "funding_stage": "Bootstrapped",
        "stack": ["Python"],
        "linkedin": "https://linkedin.com/in/bob-solo",
        "country": "United States",
    },
    "carol@megacorp.global": {
        "name": "Carol Park",
        "title": "Director of AI Platform",
        "company": "MegaCorp Global",
        "industry": "Fintech",
        "company_size": "12,000 employees",
        "funding_stage": "Public",
        "stack": ["Python", "TypeScript", "Anthropic", "self-hosted Llama", "Grafana", "OTel"],
        "linkedin": "https://linkedin.com/in/carol-park",
        "country": "United Kingdom",
    },
    "dave@langchain.com": {
        "name": "Dave Rivera",
        "title": "Head of DevRel",
        "company": "LangChain",  # competitor — should disqualify
        "industry": "AI Infrastructure",
        "company_size": "60 employees",
        "funding_stage": "Series A",
        "stack": ["Python", "TypeScript"],
        "linkedin": "https://linkedin.com/in/dave-rivera",
        "country": "United States",
    },
    "eve@neobank.io": {
        "name": "Eve Tanaka",
        "title": "Staff Engineer",
        "company": "NeoBank",
        "industry": "Fintech",
        "company_size": "350 employees",
        "funding_stage": "Series C",
        "stack": ["Python", "OpenAI", "Datadog", "AWS"],
        "linkedin": "https://linkedin.com/in/eve-tanaka",
        "country": "Japan",  # geography risk per ICP
    },
}


# ─── Pluggable enrichment backends ───────────────────────────────────────────


def _enrich_mock(email: str) -> dict | None:
    return _MOCK_LEADS.get(email.lower())


def _enrich_clearbit(email: str) -> dict | None:
    """Live Clearbit Enrichment. Requires CLEARBIT_API_KEY + httpx."""
    import httpx

    api_key = os.getenv("CLEARBIT_API_KEY")
    if not api_key:
        raise RuntimeError("ENRICHMENT_BACKEND=clearbit but CLEARBIT_API_KEY is not set.")
    response = httpx.get(
        f"https://person.clearbit.com/v2/combined/find?email={email}",
        auth=(api_key, ""),
        timeout=20.0,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()
    person = payload.get("person", {}) or {}
    company = payload.get("company", {}) or {}
    return {
        "name": person.get("name", {}).get("fullName"),
        "title": (person.get("employment") or {}).get("title"),
        "company": company.get("name"),
        "industry": company.get("category", {}).get("industry"),
        "company_size": str((company.get("metrics") or {}).get("employees")),
        "linkedin": (person.get("linkedin") or {}).get("handle"),
        "country": company.get("geo", {}).get("country"),
    }


_ENRICHMENT_BACKENDS = {"mock": _enrich_mock, "clearbit": _enrich_clearbit}


# ─── Pluggable CRM backends ──────────────────────────────────────────────────


def _crm_log_mock(payload: dict) -> dict:
    crm_id = f"CRM-{uuid.uuid4().hex[:10]}"
    log_path = _HERE / ".fastaiagent" / "crm_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps({"crm_id": crm_id, "ts": time.time(), **payload}) + "\n")
    return {"crm_id": crm_id, "logged_at": time.time()}


def _crm_log_salesforce(payload: dict) -> dict:
    """Live Salesforce REST. Requires SALESFORCE_INSTANCE_URL + SALESFORCE_ACCESS_TOKEN."""
    import httpx

    instance = os.getenv("SALESFORCE_INSTANCE_URL")
    token = os.getenv("SALESFORCE_ACCESS_TOKEN")
    if not (instance and token):
        raise RuntimeError("CRM_BACKEND=salesforce but creds missing.")
    response = httpx.post(
        f"{instance}/services/data/v59.0/sobjects/Lead/",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=20.0,
    )
    response.raise_for_status()
    return {"crm_id": response.json().get("id", ""), "logged_at": time.time()}


_CRM_BACKENDS = {"mock": _crm_log_mock, "salesforce": _crm_log_salesforce}


# ─── Pluggable email backends ────────────────────────────────────────────────


def _email_send_mock(*, to: str, subject: str, body: str) -> dict:
    msg_id = f"MSG-{uuid.uuid4().hex[:10]}"
    log_path = _HERE / ".fastaiagent" / "email_outbox.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps({"msg_id": msg_id, "to": to, "subject": subject, "body": body}) + "\n")
    return {"sent": True, "msg_id": msg_id, "ts": time.time()}


def _email_send_sendgrid(*, to: str, subject: str, body: str) -> dict:
    """Live SendGrid send. Requires SENDGRID_API_KEY + SENDGRID_FROM."""
    import httpx

    api_key = os.getenv("SENDGRID_API_KEY")
    from_addr = os.getenv("SENDGRID_FROM")
    if not (api_key and from_addr):
        raise RuntimeError("EMAIL_BACKEND=sendgrid but SENDGRID_API_KEY/SENDGRID_FROM missing.")
    response = httpx.post(
        "https://api.sendgrid.com/v3/mail/send",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "personalizations": [{"to": [{"email": to}]}],
            "from": {"email": from_addr},
            "subject": subject,
            "content": [{"type": "text/plain", "value": body}],
        },
        timeout=20.0,
    )
    response.raise_for_status()
    return {
        "sent": True,
        "msg_id": response.headers.get("X-Message-Id", ""),
        "ts": time.time(),
    }


_EMAIL_BACKENDS = {"mock": _email_send_mock, "sendgrid": _email_send_sendgrid}


# ─── Idempotent inner sender ────────────────────────────────────────────────


def _email_idem_key(*, to: str, subject: str, body: str) -> str:
    """Cache key includes the body hash so subject-only retries still
    correctly de-duplicate. Email is a hard side effect — same recipient
    + same content must yield the same msg_id across retries."""
    digest = hashlib.sha256(f"{to}|{subject}|{body}".encode()).hexdigest()[:16]
    return f"email:{digest}"


@idempotent(key_fn=_email_idem_key)
def _persist_send(*, to: str, subject: str, body: str) -> dict:
    """The actual send. Decorated so a resume-after-interrupt or a Replay
    rerun produces the same message id rather than double-sending."""
    backend = os.getenv("EMAIL_BACKEND", "mock")
    fn = _EMAIL_BACKENDS.get(backend, _email_send_mock)
    return fn(to=to, subject=subject, body=body)


# ─── Shared deps ─────────────────────────────────────────────────────────────


@dataclass
class SDRDeps:
    notes: list[str] = field(default_factory=list)


def make_deps() -> SDRDeps:
    return SDRDeps()


# ─── JSON helpers ────────────────────────────────────────────────────────────


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n|\n```\s*$")


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = _FENCE_RE.sub("", text)
    return text.strip()


def _parse_json_loose(text: str) -> dict:
    """Best-effort JSON parse — strips code fences, falls back to {} on failure."""
    if isinstance(text, dict):
        return text  # already parsed
    try:
        return json.loads(_strip_fences(text))
    except Exception:
        return {}


# ─── Lazy agent factories ────────────────────────────────────────────────────
#
# Built once on first use so module import is cheap. The score and draft
# tools wrap these — the Chain only ever sees tool nodes.

_score_agent: fa.Agent | None = None
_draft_agent: fa.Agent | None = None


def _get_score_agent() -> fa.Agent:
    global _score_agent
    if _score_agent is None:
        from workflow import SCORE_PROMPT, icp_kb_search

        _score_agent = fa.Agent(
            name="lead-scorer",
            system_prompt=SCORE_PROMPT,
            llm=fa.LLMClient(provider="openai", model=os.getenv("LLM_MODEL", "gpt-4o")),
            tools=[icp_kb_search],
        )
    return _score_agent


def _get_draft_agent() -> fa.Agent:
    global _draft_agent
    if _draft_agent is None:
        from workflow import OUTREACH_PROMPT

        _draft_agent = fa.Agent(
            name="outreach-drafter",
            system_prompt=OUTREACH_PROMPT,
            llm=fa.LLMClient(provider="openai", model=os.getenv("LLM_MODEL", "gpt-4o")),
        )
    return _draft_agent


# ─── Tools used as Chain nodes ───────────────────────────────────────────────


@fa.tool()
def enrich_lead(prospect_email: str, ctx: fa.RunContext[SDRDeps]) -> dict:
    """Fetch firmographic + technographic data for a prospect by email.
    Returns a dict that becomes the chain's ``state.output``; downstream
    nodes pull individual fields via ``{{node_results.enrich.output.<key>}}``
    templates."""
    backend = os.getenv("ENRICHMENT_BACKEND", "mock")
    fn = _ENRICHMENT_BACKENDS.get(backend, _enrich_mock)
    record = fn(prospect_email)
    if record is None:
        return {"found": False, "email": prospect_email, "reason": "no enrichment record"}
    return {"found": True, "email": prospect_email, **record}


@fa.tool()
async def score_lead(enriched_json: str, ctx: fa.RunContext[SDRDeps]) -> dict:
    """Run the score-agent against enriched lead data. Parses the agent's
    JSON output. Returns ``{"score": float, "qualified": bool, "reasons": list[str]}``.

    The score-agent itself uses ``icp_kb_search`` to read the ICP playbook
    before scoring — the playbook lives in ``knowledge/icp_playbook.md``
    and is hot-editable without re-ingest."""
    agent = _get_score_agent()
    result = await agent.arun(enriched_json, context=ctx)
    parsed = _parse_json_loose(result.output)
    return {
        "score": float(parsed.get("score", 0.0)),
        "qualified": bool(parsed.get("qualified", False)),
        "reasons": list(parsed.get("reasons", [])),
        "raw": result.output,
    }


@fa.tool()
async def draft_outreach(
    prospect_json: str,
    score_json: str,
    ctx: fa.RunContext[SDRDeps],
) -> dict:
    """Run the outreach-drafter against the enriched prospect + score.
    Returns ``{"to": str, "subject": str, "body": str}``."""
    agent = _get_draft_agent()
    bundled = json.dumps({"prospect": _parse_json_loose(prospect_json), "score": _parse_json_loose(score_json)})
    result = await agent.arun(bundled, context=ctx)
    parsed = _parse_json_loose(result.output)
    return {
        "to": str(parsed.get("to", "")),
        "subject": str(parsed.get("subject", "")),
        "body": str(parsed.get("body", "")),
        "raw": result.output,
    }


@fa.tool()
def send_outreach_email(
    to: str,
    subject: str,
    body: str,
    ctx: fa.RunContext[SDRDeps],
) -> dict:
    """Send the outreach email — pauses for human approval first.

    The chain executor catches the ``InterruptSignal`` raised by ``interrupt()``,
    persists a checkpoint, and returns ``ChainResult(status="paused")``. The
    caller resumes via ``chain.aresume(execution_id, resume_value=fa.Resume(...))``
    and this tool re-fires with the human's decision in scope.
    """
    decision = fa.interrupt(
        reason="approve_outreach",
        context={
            "to": to,
            "subject": subject,
            "preview": body[:280] + ("..." if len(body) > 280 else ""),
        },
    )
    if not decision.approved:
        return {
            "sent": False,
            "reason": "declined",
            "approver": decision.metadata.get("approver", "unknown"),
            "notes": decision.metadata.get("notes", ""),
        }
    return _persist_send(to=to, subject=subject, body=body)


@fa.tool()
def log_outreach_to_crm(
    prospect_email: str,
    msg_id: str,
    ctx: fa.RunContext[SDRDeps],
) -> dict:
    """Record the sent outreach against the prospect's CRM record."""
    backend = os.getenv("CRM_BACKEND", "mock")
    fn = _CRM_BACKENDS.get(backend, _crm_log_mock)
    return fn(
        {
            "kind": "outreach_sent",
            "email": prospect_email,
            "msg_id": msg_id,
            "stage": "Engaged",
        }
    )


@fa.tool()
def disqualify_and_log(
    prospect_email: str,
    reason: str,
    ctx: fa.RunContext[SDRDeps],
) -> dict:
    """Log a disqualified prospect to the CRM with the rejection reason."""
    backend = os.getenv("CRM_BACKEND", "mock")
    fn = _CRM_BACKENDS.get(backend, _crm_log_mock)
    return fn(
        {
            "kind": "disqualify",
            "email": prospect_email,
            "reason": reason,
            "stage": "Disqualified",
        }
    )
