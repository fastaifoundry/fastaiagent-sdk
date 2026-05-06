# Sales SDR Agent (Chain DAG with HITL)

A Clay / HubSpot Breeze / Salesforce Agentforce-style outbound sales agent built with [FastAIAgent SDK](https://github.com/fastaifoundry/fastaiagent-sdk) v1.6.1+. A `Chain` DAG orchestrates the pipeline; the **verifier-style** branching lives on the *edge condition*, not the supervisor's LLM (compare to `examples/research-agent/`'s Supervisor pattern). A human approves every outreach send before it fires.

```
input: {"prospect_email": "..."}
                │
                ▼
        ┌───────────────┐
        │   enrich      │ tool — pluggable backend (mock / Clearbit / Brave)
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │   score       │ tool — wraps an LLM agent that consults the ICP
        │               │        playbook (LocalKB) and returns
        │               │        {score, qualified, reasons}
        └───────┬───────┘
                ▼
        score >= 0.7 ?
        ┌───────┴────────┐
        ▼ qualified      ▼ disqualified
   ┌──────────┐     ┌────────────────┐
   │  draft   │     │  disqualify    │ tool — log to CRM
   │  (tool   │     │  and return    │
   │ wrapping │     └────────────────┘
   │  agent)  │
   └────┬─────┘
        ▼
   ┌──────────┐  HITL: send tool calls fa.interrupt()
   │   send   │  → chain pauses → human approves →
   │          │     @idempotent _persist_send fires
   └────┬─────┘
        ▼
   ┌──────────┐
   │ log_crm  │ tool — pluggable backend (mock / Salesforce)
   └──────────┘
```

**What this example demonstrates** (vs. the prior two templates):

- `Chain` DAG with **conditional edge routing** — `chain.connect(src, dst, condition="{{state.output.score}} >= 0.7")`
- `chain.aexecute(... context=ctx)` — `RunContext` propagation through every chain node (v1.6.1+, see [docs/chains/index.md](../../docs/chains/index.md))
- LLM agents wrapped *inside* tool functions so the chain executor's state-flow stays uniform — each tool returns a typed dict the next node addresses via `{{node_results.<id>.output.<key>}}` templates
- `fa.interrupt()` inside a tool node — chain suspends, REPL approves, `chain.aresume()` re-fires
- `@idempotent` on the inner `_persist_send` keyed on `(to, subject, body)` so resume-after-approval and Replay reruns reuse the same `msg_id` instead of double-sending
- Three pluggable provider backends — enrichment (Clearbit), CRM (Salesforce), email (SendGrid) — each with a real httpx implementation included; flip via env var
- KB-backed scoring rubric — edit `knowledge/icp_playbook.md`, score-agent reads it on every run via `icp_kb_search`

---

## Quick Start

```bash
# from the SDK root
pip install -e .
cd examples/sales-sdr-agent
cp .env.example .env       # only OPENAI_API_KEY is required
pip install -r requirements.txt

python agent.py                                          # default prospect
python agent.py --prospect carol@megacorp.global
```

Ships ready to run offline — all three backends default to `mock` and the mock
prospect corpus is hard-coded in `tools.py`. Real provider stubs (Clearbit,
Salesforce, SendGrid) are wired and one env var away from being live.

---

## Files

```
sales-sdr-agent/
├── README.md
├── .env.example
├── requirements.txt
├── tools.py             # all tool nodes — enrichment / score / draft / send / log / disqualify
├── workflow.py          # Chain DAG + score / draft system prompts + ICP KB
├── agent.py             # CLI entry point with HITL approval loop
├── streaming_demo.py    # tail the trace store as the chain runs
├── replay_demo.py       # Replay.fork_at(...).rerun() at any node boundary
├── eval_suite.py        # custom Python scorers (no LLM-judge needed)
├── knowledge/
│   └── icp_playbook.md  # ICP rubric — hot-editable; KB re-loads on every run
└── tests/
    └── test_smoke.py    # 9 offline regression tests
```

---

## How it's wired

### Chain construction ([workflow.py](workflow.py))

```python
import fastaiagent as fa
from fastaiagent.chain.node import NodeType

chain = fa.Chain("sales-sdr", checkpoint_enabled=True)

chain.add_node("enrich",     type=NodeType.tool, tool=enrich_lead,
               input_mapping={"prospect_email": "{{state.prospect_email}}"})
chain.add_node("score",      type=NodeType.tool, tool=score_lead,
               input_mapping={"enriched_json": "{{state.output}}"})
chain.add_node("draft",      type=NodeType.tool, tool=draft_outreach,
               input_mapping={
                   "prospect_json": "{{node_results.enrich.output}}",
                   "score_json":    "{{node_results.score.output}}",
               })
chain.add_node("send",       type=NodeType.tool, tool=send_outreach_email,
               input_mapping={
                   "to":      "{{node_results.enrich.output.email}}",
                   "subject": "{{node_results.draft.output.subject}}",
                   "body":    "{{node_results.draft.output.body}}",
               })
chain.add_node("log_crm",    type=NodeType.tool, tool=log_outreach_to_crm, ...)
chain.add_node("disqualify", type=NodeType.tool, tool=disqualify_and_log,  ...)

# Edges — conditional routing on score
chain.connect("enrich", "score")
chain.connect("score",  "draft",      condition="{{state.output.score}} >= 0.7", label="qualified")
chain.connect("score",  "disqualify", condition="{{state.output.score}} < 0.7",  label="disqualified")
chain.connect("draft",  "send")
chain.connect("send",   "log_crm")
```

### Scoring as a tool that wraps an Agent ([tools.py](tools.py))

The score and draft "agents" don't sit on agent nodes — they're wrapped inside
tool functions so the chain's state-flow is uniform. Each tool returns a
typed dict that the next step addresses via dotted templates:

```python
@fa.tool()
async def score_lead(enriched_json: str, ctx: fa.RunContext[SDRDeps]) -> dict:
    """Run the score-agent against enriched data; parse JSON output."""
    agent = _get_score_agent()       # lazy singleton
    result = await agent.arun(enriched_json, context=ctx)
    parsed = _parse_json_loose(result.output)
    return {
        "score":     float(parsed.get("score", 0.0)),
        "qualified": bool(parsed.get("qualified", False)),
        "reasons":   list(parsed.get("reasons", [])),
        "raw":       result.output,
    }
```

### HITL approval gate ([tools.py](tools.py))

```python
@fa.tool()
def send_outreach_email(to, subject, body, ctx) -> dict:
    decision = fa.interrupt(
        reason="approve_outreach",
        context={"to": to, "subject": subject, "preview": body[:280]},
    )
    if not decision.approved:
        return {"sent": False, "reason": "declined", ...}
    return _persist_send(to=to, subject=subject, body=body)  # @idempotent
```

The chain executor catches the `InterruptSignal`, persists a checkpoint, and
returns `ChainResult(status="paused")`. The CLI's resume loop:

```python
while result.status == "paused":
    info = result.pending_interrupt
    answer = input("  Approve send? [y/N]: ").strip().lower()
    result = await chain.aresume(
        execution_id,
        resume_value=fa.Resume(approved=(answer in {"y", "yes"})),
        context=ctx,
    )
```

### Idempotent send ([tools.py](tools.py))

```python
@idempotent(key_fn=_email_idem_key)
def _persist_send(*, to, subject, body) -> dict:
    backend = os.getenv("EMAIL_BACKEND", "mock")
    return _EMAIL_BACKENDS[backend](to=to, subject=subject, body=body)
```

The cache key is `sha256(to + subject + body)`. After the human approves, the
chain re-fires `send_outreach_email`; the inner `_persist_send` returns the
cached receipt instead of sending twice. Same protection on a `Replay.rerun`.

### Pluggable backends ([tools.py](tools.py))

| Backend | Default | Real stub | Env vars |
|---|---|---|---|
| `ENRICHMENT_BACKEND` | `mock` (5 hard-coded prospects) | `_enrich_clearbit` (Clearbit Enrichment) | `CLEARBIT_API_KEY` |
| `CRM_BACKEND` | `mock` (writes JSONL to `.fastaiagent/crm_log.jsonl`) | `_crm_log_salesforce` (Salesforce REST) | `SALESFORCE_INSTANCE_URL`, `SALESFORCE_ACCESS_TOKEN` |
| `EMAIL_BACKEND` | `mock` (writes JSONL to `.fastaiagent/email_outbox.jsonl`) | `_email_send_sendgrid` (SendGrid v3) | `SENDGRID_API_KEY`, `SENDGRID_FROM` |

Each real stub uses `httpx` (declared optional in `requirements.txt`) and includes the provider's URL + field mapping. Set the corresponding env var, uncomment `httpx` in requirements.txt, and you're live.

---

## Running each entry point

```bash
# Single prospect (default: alice@acme-saas.com)
python agent.py
python agent.py --prospect carol@megacorp.global   # too big — disqualifies
python agent.py --prospect dave@langchain.com      # competitor — disqualifies

# Stream the chain trace as it executes
python streaming_demo.py --prospect carol@megacorp.global

# Replay debugging — fork at a node boundary, swap a field, rerun
python replay_demo.py

# Eval suite — 4 cases × 3 custom scorers (scoring_correct,
# outreach_personalized, idempotent_send)
python eval_suite.py

# Smoke tests — 9 offline tests, ~0.5s
python -m pytest tests/
```

---

## Local UI

```bash
fastaiagent ui start             # http://127.0.0.1:7842
```

What this example populates:

- **`/traces`** — the chain run lands as a `chain.sales-sdr` root span. Inside it: `tool.enrich_lead`, then nested `agent.lead-scorer` (with its `tool.icp_kb_search` and an `llm.openai.gpt-4o` span), `agent.outreach-drafter`, `tool.send_outreach_email`, `tool.log_outreach_to_crm`. Disqualified runs follow a different path through `tool.disqualify_and_log`.
- **`/agents`** — dependency graph shows the `lead-scorer` and `outreach-drafter` agents, each with their tool inventories.
- **`/evals`** — `eval_suite.py` persists each run as `sales-sdr eval` against dataset `sdr-prospects-golden`.
- **`/approvals`** (when a chain pauses) — the pending interrupt row appears here with the email preview; click Approve / Decline to drive `chain.aresume()` from the UI.

---

## Customising

**Edit the ICP rubric** — Open `knowledge/icp_playbook.md` and rewrite the firmographic / technographic / disqualifier sections. The score-agent re-reads it via `icp_kb_search` on every run; no rebuild step.

**Tighten the qualified threshold** — `QUALIFIED_THRESHOLD=0.85` in `.env` (default `0.7`). Both edge conditions read the same env var.

**Add another node** (e.g., a "warm-up" step that posts a LinkedIn comment before the email):

```python
chain.add_node("warm_up", type=NodeType.tool, tool=post_linkedin_comment, ...)
chain.connect("draft",   "warm_up")
chain.connect("warm_up", "send")
```

**Switch to real Clearbit + Salesforce + SendGrid** — Uncomment `httpx` in `requirements.txt`, set the three API-key env vars, and flip `*_BACKEND=clearbit / salesforce / sendgrid` in `.env`. No code change.

---

## What this example does NOT demonstrate

- **Single-agent** workflows (memory, simple HITL, KB-grounded support) — see `examples/customer-support-agent/`.
- **LLM-driven multi-agent orchestration** with revision loops — see `examples/research-agent/` (Supervisor pattern).
- **Multi-turn chat memory** — this example is single-shot per prospect.
- **Multimodal inputs** — text-only.
- **Custom Pydantic `output_type`** — the score-agent and drafter both produce JSON via system-prompt instructions, parsed in the wrapping tool. For a strict-schema variant, set `output_type=ScoreReport` on the score-agent.

---

## License

Apache 2.0 — same as the SDK.
