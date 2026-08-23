# Connected eval export (Agent-CI verdicts)

[Agent CI](../evaluation/agent-ci.md) gates your build locally and works fully
standalone — no platform dependency. When you `fa.connect()` to an **Enterprise
control plane**, the SDK additionally **reports each gated run's verdict** so the
org can answer "which of our agents passed their gates this week?" and treat an
agent that produces no eval evidence as a governance finding rather than a blank
space.

## Mark, don't coerce

Eval export is **egress**, not enforcement, and the SDK keeps those in separate
lanes:

| | Who decides | Local off-switch |
|---|---|---|
| **Enforcement** — plane-authored guardrails | plane authors, edge enforces | no (by design) |
| **Egress** — traces, eval verdicts | your runtime | **yes, and it is final** |

`connect(export_evals=False)` is honored, always. The plane cannot override it,
exactly as with `export_traces`. What a connected plane does instead is *mark*
agents that produce no eval evidence, and it can refuse to promote an agent that
cannot produce any — you are never blocked locally, and nothing is forced out of
your machine.

Because the SDK attests its posture at enroll, the plane distinguishes two states
that otherwise look identical:

- **"eval export disabled"** — you run evals and keep them local (a deliberate choice)
- **"no eval data in N days"** — no evals are running at all

## What is reported

**Metadata only.** Run aggregates, the gate verdict, thresholds, git provenance,
and per-case scorer verdicts travel. Case **inputs, expected outputs and actual
outputs never do** — the plane joins content through each case's `trace_id`
against traces it already ingested, so a verdict can be opened against real
content without shipping a second copy of it.

Preview the literal payload before anything leaves:

```bash
fastaiagent eval export --dry-run     # the exact JSON
fastaiagent eval export --status      # posture + how many runs are queued
```

## Wire protocol

`POST {target}/public/v1/eval/runs/ingest` · headers `X-API-Key`,
`Content-Type: application/json` · **wire v1.6**

Batched, at-least-once, **idempotent on `run_id`**. The response counts new runs
only, so a re-send is free:

```json
{ "ingested": 2 }
```

### Request

```json
{
  "runs": [
    {
      "run_id": "76fd558996c54dc598b5d0652d5dbbfc",
      "run_name": "pytest::northwind-support",
      "dataset_name": "cases.jsonl",
      "agent_name": "northwind-support",
      "started_at": "2026-08-23T09:14:02.511034+00:00",
      "finished_at": "2026-08-23T09:14:19.884120+00:00",
      "pass_count": 18,
      "fail_count": 2,
      "pass_rate": 0.9,
      "errored_count": 1,
      "error_rate": 0.0476,
      "gate_outcome": "failed",
      "thresholds": { "overall.pass_rate": 0.95 },
      "scorers": ["exact_match", "faithfulness"],
      "git_sha": "9f3c1ab7e2d4c5f6a8b9c0d1e2f3a4b5c6d7e8f9",
      "git_branch": "feature/refund-policy",
      "baseline": {
        "run_id": "d41c9b2e77a34f0e8b6c1a5d3e9f7b28",
        "pass_rate_delta": -0.05,
        "regressed_count": 2
      },
      "sdk_version": "1.49.0",
      "instance_id": "be42fb1d3f7847e98e0ace31b2a05f40",
      "cases": [
        {
          "case_id": "0b1caf35e17c45d5baeb0fde33551e39",
          "ordinal": 0,
          "per_scorer": {
            "exact_match": { "passed": true, "score": 1.0, "reason": null }
          },
          "trace_id": "ae077737d85c4f2b9e1a6c3d5f8b0a2c",
          "error": null
        },
        {
          "case_id": "7c2e9f11ab3d40a6b8e5d4c7f1a92b60",
          "ordinal": 1,
          "per_scorer": {},
          "trace_id": null,
          "error": "provider 500"
        }
      ]
    }
  ]
}
```

### Run fields

| Field | Type | Meaning |
|---|---|---|
| `run_id` | string | **idempotency key** — SDK-generated; the plane dedupes on it |
| `run_name` | string? | e.g. `pytest::<rootdir>`, or `--eval-run-name` |
| `dataset_name` | string? | the dataset the cases came from |
| `agent_name` | string? | **resolved to a real plane agent at ingest** — this is what makes the run *evidence for a specific system*. Inferred from the callable (`agent.run` → `Agent.name`); `null` when a suite spans several agents |
| `started_at` / `finished_at` | ISO 8601? | evidence freshness / control decay |
| `pass_count` / `fail_count` / `pass_rate` | int/int/float? | the scored result |
| `errored_count` / `error_rate` | int/float? | infra failures — **unscored**, never counted as quality misses |
| `gate_outcome` | `passed`\|`failed`\|`invalid` | **required**. `invalid` = infra disqualified the run |
| `thresholds` | object? | what the gate demanded, keyed `<metric>.<field>`. `{}` = evals ran, no gate demanded |
| `scorers` | string[]? | scorer names used |
| `git_sha` / `git_branch` | string? | **which version was tested** |
| `baseline` | object? | `{run_id, pass_rate_delta, regressed_count}` — evidences *regressions tracked across versions*, not merely "evals ran" |
| `sdk_version` / `instance_id` | string? | provenance; `instance_id` joins the enrollment record |
| `cases` | array | per-case verdicts (below) |

### Case fields

| Field | Type | Meaning |
|---|---|---|
| `case_id` | string? | SDK-generated; the row key |
| `ordinal` | int? | position in the dataset |
| `per_scorer` | object? | `{scorer: {passed, score, reason}}`. Empty for an errored case |
| `trace_id` | string? | **the join key** — lets the plane corroborate this verdict against a trace it ingested independently |
| `error` | string? | infra failure detail; presence means the case was never scored |

!!! warning "Never on the wire"
    `input`, `expected_output`, `actual_output`. A test freezes the exact payload
    shape and asserts these three are absent — adding them later is a privacy
    regression, not an enhancement.

    Note that `run_name`, `dataset_name` and especially `git_branch` *do* travel,
    and branch names routinely carry ticket IDs or customer names.

### Enrollment attestation

The posture rides along on the existing enroll call so the plane can tell a
deliberate opt-out from an absence of evals:

```json
POST /public/v1/governance/enroll
{
  "instance_id": "be42fb1d3f7847e98e0ace31b2a05f40",
  "sdk_version": "1.49.0",
  "fail_mode": "open",
  "protocol_version": "1",
  "export_evals": true
}
```

### Errors

| Status | Meaning | SDK behavior |
|---|---|---|
| 2xx | ingested | mark runs synced |
| 403 | key lacks **`eval:execute`**, or domain lacks **`connected_state_plane`** | terminal — warn once, leave buffered to age out |
| 404 | plane predates wire v1.6 | terminal — warn once |
| 5xx / transport | transient | retry ×3 with 0.5s, 1.0s backoff, then leave buffered |

`eval:execute` is **not** a default scope; mint keys with it explicitly.

## Delivery model

Runs are queued in `local.db` (`eval_runs.synced = 0`) and pushed by a background
drain. A run flips to `synced = 1` **only after a confirmed 2xx** — so an outage
simply buffers, and the next gated run drains the backlog. The buffer is bounded
(50k runs / 30 days); beyond that the oldest un-acked runs are *abandoned*, which
marks them handled without deleting anything — they stay in `local.db` and the
Local UI.

A run becomes exportable when its **gate verdict** is recorded, not when it is
persisted: the plane requires `gate_outcome`, and the gate necessarily runs after
persistence. Runs nobody gated still export with `thresholds: {}` — "evals ran,
no gate demanded" is evidence too.

When export is off, runs are written `synced = 1` and never queued, so a disabled
install never grows an outbox it cannot drain. Turning it on later ships only new
runs, never a backlog dump.

## See also

- [Agent CI](../evaluation/agent-ci.md) — the gates themselves
- [Connected HITL](connected-hitl.md) — the same observer model for pauses
- [Connected governance](connected-governance.md) — enrollment and fail-mode
