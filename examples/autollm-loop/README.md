# AutoLLM closed loop — from bad traces to a better production prompt

Your triage agent is live. Its prompt is in the registry, and every ticket it
handles is traced. Some of its answers are wrong. This example turns those
traces into a labelled dataset, lets [AutoLLM](../../docs/evaluation/optimization.md)
write the next prompt, registers it as a new version, gates it like any other
change, and ships it by moving one alias. No mocks: it runs against the OpenAI
API, and every step shows up in the Local UI.

Requires `fastaiagent>=1.87.0`. Docs page: [AutoLLM Closed Loop](../../docs/flagships/autollm-closed-loop.md).

## Run it

```sh
cd examples/autollm-loop
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
./run_all.sh            # ~10 minutes, from a clean slate
fastaiagent ui          # from this folder — it reads ./.fastaiagent/local.db
```

Or one step at a time — they share `./.fastaiagent/local.db`, so run them from
this folder:

| Step | Script | What it does | Open in the UI |
|---|---|---|---|
| 1 | `01_register_prompt.py` | registers v1 of `ticket-triage`, points `production` at it | Prompts |
| 2 | `02_serve_traffic.py` | 120 tickets through the `production` prompt — each a trace stamped with the prompt version | Traces · Prompts → lineage |
| 3 | `03_curate.py` | `curate_from_traces` → the support leads' labels → the Dataset Editor's file | Datasets → ticket-triage |
| 4 | `04_baseline.py` | scores v1 on the dataset — the baseline the gate compares against | Eval Runs |
| 5 | `05_optimize.py` | AutoLLM: prompt rewrites + worked examples, scored on dev, guarded by a holdout | AutoLLM |
| 6 | `06_promote.py` | the winner becomes v2 — template, examples, provenance; `production` doesn't move | Prompts (diff) |
| 7 | `07_go_live.py` | runs `test_gate.py` as CI would; green → `production` moves to v2 | Eval Runs → compare |
| 2 | `02_serve_traffic.py` again | the same traffic, now on v2 | Prompts → lineage |

`triage.py` holds what every step shares: `V1_PROMPT`, the `TriageMatch` scorer
and `load_agent()`, which always builds the agent **from the registry**.

## The house rules

v1 names the queues and the format. These rules aren't in it — they live only
in the labels, and the loop's job is to get them into the prompt:

- a billing dispute **over EUR 500** is P1; at or under, P3
- locked out / password / 2FA → `account` P2, not `security`
- unrecognised login, leaked key, unknown admin → `security` P1
- an outage is **P1 on Enterprise**, P2 on Free or Team
- a delivery **5+ business days** late is P2; less, P3
- a GDPR export or erasure request → `account` P2

The tickets come in matched pairs that differ only in the deciding fact
(Enterprise / Team, EUR 480 / EUR 520, 4 days / 5 days).

## A real run

From one `./run_all.sh` (`gpt-4.1-mini` agent, `gpt-5` proposer), trimmed.

**Steps 3–4** — production got 49 of 120 wrong; re-scored, v1 passes 56%, and
every miss is a priority:

```
labelled 120 cases → .fastaiagent/datasets/ticket-triage.jsonl
production got 49 of 120 wrong by the support leads' labels.
triage_match: avg=0.78 pass_rate=56% (120 cases)

what it gets wrong:
  19  priority P1, expected P2
  15  priority P2, expected P3
  12  priority P2, expected P1
   7  priority P3, expected P2
```

**Step 5** — AutoLLM alternates the two levers; the holdout (30 tickets no step
of the search saw) goes from 0.633 to 0.967:

```
Optimization — ticket-triage (stopped: patience)
baseline   dev=0.567
 iter 1 [instructions]  dev=0.767 (+0.200)  ACCEPT  — Adds explicit, concise priority rules …
 iter 2 [fewshot]  dev=0.833 (+0.267)  ACCEPT  — few-shot k=4
 iter 3 [instructions]  dev=0.933 (+0.367)  ACCEPT  — Introduces a brief 'Escalation overrides' section …
 iter 4 [fewshot]  dev=0.967 (+0.400)  ACCEPT  — few-shot k=2
 (rejected candidates omitted)
best        dev=0.967
holdout     best=0.967 (baseline=0.633, Δ+0.333) → winner kept
```

**Step 6** — v2 is v1 plus the rules it had been missing (excerpt):

```
+Escalation overrides (apply before other rules):
+- Technical: Enterprise plan/contract/workspace + production-wide failure … -> technical P1.
+- Shipping: hardware security key shipment overdue by ≥5 business/working days … -> shipping P2.
+- Billing monetary disputes or refund/charge >= 500 (EUR or equivalent) -> billing P1.
+- Other billing issues (… refunds < 500) -> billing P3.
+- Technical service problems (…) -> technical P2 (do not use P1 unless an escalation applies).
+- Account access issues (2FA problems, lost authenticator, …) -> account P2.
+- Privacy/data requests (GDPR access/erasure/portability) -> account P2.
```

**Step 7** — the gate, case by case against v1:

```
overall pass_rate=96%
overall.pass_rate: 0.9583 >= 0.85 required
baseline: ticket-triage v1 pass_rate=0.5583
current:  (current session) pass_rate=0.9583 (delta +0.4000)
regressed=3 improved=51 unchanged_pass=64 unchanged_fail=2
  regressed: "What's the rate limit on the search endpoint?" (scorers: triage_match)
  regressed: "Does the API support idempotency keys for POST requests?" (scorers: triage_match)
  regressed: "Can you send our invoices to finance@ourcompany.example instead of to me?" …

gate passed — 'production' now points at v2.
```

The three regressions are named, not hidden: two how-to questions v2 now rates
P2, and an invoice-address change it routes to `account`. Whether that trade is
worth +40 points is a reviewer's call — the gate's job is to put it in front of
one. The whole loop made 1,202 model calls for about **$0.30**.

## Notes

- **Models.** The agent runs on `gpt-4.1-mini`; the prompts are written by
  `gpt-5` (`proposer_llm`). Inferring a house rule from labelled failures is
  induction — a reasoning model does it far better. Override with
  `TRIAGE_MODEL` and `TRIAGE_PROPOSER_MODEL`.
- **The gate is a regression check.** It scores all 120 tickets, including
  AutoLLM's training split. The generalisation number is step 5's holdout.
- **Labels.** Starring traces is a UI action, so step 3 applies labels from
  `data/tickets.jsonl` and the loop runs unattended. Edit a label in the
  Dataset Editor and step 4 onward uses your edit.
- **Scores vary.** Seeded split, non-deterministic model: across our runs the
  holdout landed between 0.90 and 0.97, and one run's winner over-generalised
  ("any full outage is P1") — the holdout is how you see that.
- **Start over.** `./run_all.sh` deletes this folder's `.fastaiagent/` and
  `out/` first. `python 01_register_prompt.py --reset` resets only the prompt,
  the dataset and `out/`.
- **Offline tests.** `pytest tests/` checks the data, the scorer and
  `load_agent` without calling a model.
