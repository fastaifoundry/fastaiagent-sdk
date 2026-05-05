"""
Workflow — the Chain DAG that orchestrates the SDR pipeline.

```
input: {"prospect_email": "..."}
                │
                ▼
        ┌───────────────┐
        │   enrich      │ tool — _enrich_lead
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │   score       │ tool — wraps score-agent (KB-grounded ICP rubric)
        └───────┬───────┘
                ▼
        score >= 0.7 ?
        ┌───────┴────────┐
        ▼ qualified      ▼ disqualified
   ┌──────────┐     ┌────────────────┐
   │  draft   │     │  disqualify    │ tool — log to CRM
   │ (agent)  │     └────────────────┘
   └────┬─────┘
        ▼
   ┌──────────┐  HITL: send tool calls interrupt()
   │   send   │  → chain pauses → human approves → @idempotent send
   └────┬─────┘
        ▼
   ┌──────────┐
   │ log_crm  │
   └──────────┘
```

Every node is a tool. The "agent" steps (score, draft) wrap LLM agents
*inside* their tool functions and return parsed JSON dicts — this keeps
the chain executor's state-flow uniform: each tool's return becomes
``state.output`` and is also addressable via ``node_results.<id>.output.<key>``.

The HITL pause lives in ``send_outreach_email`` (tools.py); the
score → draft / disqualify branch lives on the edge condition.
"""

from __future__ import annotations

import os
from pathlib import Path

import fastaiagent as fa
from fastaiagent.chain.node import NodeType

from tools import (
    SDRDeps,
    disqualify_and_log,
    draft_outreach,
    enrich_lead,
    log_outreach_to_crm,
    score_lead,
    send_outreach_email,
)

_HERE = Path(__file__).resolve().parent

# ─── Knowledge base — the ICP playbook ───────────────────────────────────────

icp_kb = fa.LocalKB(
    name="icp-playbook",
    path=str(_HERE / ".fastaiagent-kb"),
    chunk_size=512,
    chunk_overlap=50,
)
if icp_kb.status()["chunk_count"] == 0:
    icp_kb.add(str(_HERE / "knowledge"))


@fa.tool()
def icp_kb_search(query: str, ctx: fa.RunContext[SDRDeps]) -> str:
    """Look up the ICP playbook for relevant rules. Used by the score agent."""
    results = icp_kb.search(query, top_k=5)
    if not results:
        return "(no playbook entries matched)"
    return "\n\n".join(r.chunk.content for r in results)


# ─── Worker prompts (used by tool-wrapped agents in tools.py) ────────────────

SCORE_PROMPT = """You are the lead-qualification node in a sales pipeline.

Input: a JSON blob describing one prospect (firmographic + technographic data
returned by the enrichment node). It will arrive as text in your user message.

Workflow:
  1. Call ``icp_kb_search`` (with a relevant query) at least once to load
     the relevant ICP rules.
  2. Score the prospect against the rubric on a 0.0–1.0 float scale.
  3. Return ONLY valid JSON of this shape:
        {"score": <float 0-1>, "qualified": <bool>, "reasons": ["...", "..."]}
     Set ``qualified`` to true iff ``score >= 0.7``.

Rules:
  * Honor every disqualifier in the playbook (competitors, hobbyists,
    foundation-model companies, etc.) — set score < 0.5 and qualified=false.
  * Cite specific reasons. Reviewers downstream rely on them.
  * NEVER invent firmographic data the enrichment didn't supply.
  * Return RAW JSON — no Markdown fences, no commentary.
"""


OUTREACH_PROMPT = """You are the outreach-drafter node in a sales pipeline.

Input: a JSON blob with two top-level keys:
  * ``prospect``: the enriched record (name, title, company, stack, country, ...)
  * ``score``: the scorer's output ({score, qualified, reasons})

Compose a short personalized email — 80 to 130 words — that:
  * Opens with one specific reference to their stack or context (no generic "I noticed").
  * States our value proposition in one sentence.
  * Proposes a low-friction next step (15-min call, async demo link).

Return ONLY valid JSON of this shape (no Markdown fences, no commentary):
    {"to": "<prospect.email>", "subject": "...", "body": "..."}

Rules:
  * Keep ``subject`` under 60 characters.
  * No emoji, no all-caps, no marketing fluff.
  * If the scorer's reasons mention something specific (e.g. "uses Datadog"),
    the email body MUST hook into it. Reviewers will reject generic copy.
"""


# ─── Build the Chain ─────────────────────────────────────────────────────────


def build_chain() -> fa.Chain:
    """Construct the SDR pipeline. Re-runnable; safe to call from tests."""
    chain = fa.Chain("sales-sdr", checkpoint_enabled=True)

    chain.add_node(
        "enrich",
        type=NodeType.tool,
        tool=enrich_lead,
        input_mapping={"prospect_email": "{{state.prospect_email}}"},
    )

    chain.add_node(
        "score",
        type=NodeType.tool,
        tool=score_lead,
        # The previous tool's output (a dict) is in state.output and gets
        # JSON-stringified by the template renderer when the value is non-string.
        input_mapping={"enriched_json": "{{state.output}}"},
    )

    chain.add_node(
        "draft",
        type=NodeType.tool,
        tool=draft_outreach,
        # Pull both bundles from node_results so the draft sees the full
        # enrichment + the score's reasoning, not just the score dict.
        input_mapping={
            "prospect_json": "{{node_results.enrich.output}}",
            "score_json": "{{node_results.score.output}}",
        },
    )

    chain.add_node(
        "send",
        type=NodeType.tool,
        tool=send_outreach_email,
        input_mapping={
            "to": "{{node_results.enrich.output.email}}",
            "subject": "{{node_results.draft.output.subject}}",
            "body": "{{node_results.draft.output.body}}",
        },
    )

    chain.add_node(
        "log_crm",
        type=NodeType.tool,
        tool=log_outreach_to_crm,
        input_mapping={
            "prospect_email": "{{node_results.enrich.output.email}}",
            "msg_id": "{{node_results.send.output.msg_id}}",
        },
    )

    chain.add_node(
        "disqualify",
        type=NodeType.tool,
        tool=disqualify_and_log,
        input_mapping={
            "prospect_email": "{{state.prospect_email}}",
            "reason": "{{node_results.score.output.reasons}}",
        },
    )

    # ── Edges ────────────────────────────────────────────────────────────────
    threshold = float(os.getenv("QUALIFIED_THRESHOLD", "0.7"))
    chain.connect("enrich", "score")
    # Branch on the score node's output. The condition expression renders
    # {{state.output.score}} → a string ("0.85") and compares as float.
    chain.connect(
        "score",
        "draft",
        condition=f"{{{{state.output.score}}}} >= {threshold}",
        label="qualified",
    )
    chain.connect(
        "score",
        "disqualify",
        condition=f"{{{{state.output.score}}}} < {threshold}",
        label="disqualified",
    )
    chain.connect("draft", "send")
    chain.connect("send", "log_crm")

    return chain
