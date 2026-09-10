# CLAUDE.md — fastaiagent-sdk

Guidance for working in this repo. The control plane
(`fastaiagent-enterprise`) has had its own `CLAUDE.md` for a long time; this one
did not exist until 1.62.0, which is the mechanical reason the cross-repo rules
below kept being re-derived — and occasionally re-derived wrong.

---

## 1. The split — what this repo is

**The SDK executes. The plane authors, distributes, monitors and flags.**

The SDK is the enforcement point (PEP): a guardrail block always happens locally,
inside the customer's process. The plane is the policy authority and the evidence
sink, never a kill-switch. The one exception is the hosted-MCP `tools/call`
boundary, where the plane *is* the executor.

**Asymmetry between the two is normal and usually correct.** Do not file
plane-only capability as an SDK gap, or vice versa:

| Plane-only, by design | SDK-only, by design |
|---|---|
| Authoring, the template catalogue, `floor` governance, draft/publish | `reask` — the plane runs no agent loop |
| Analytics, monitors, alerting, `/guardrails/observed` | `fa.guardrail_context()` — no runtime to hold a ContextVar centrally |
| The EU-AI-Act compliance derivation | `timeout_seconds`, the ReDoS bound |
| | `code` guardrails (`fn=`), refused centrally for RCE |

**The only contract that binds the two:** a rule the plane distributes must be
**executed faithfully at the edge** — same verdict, same outcome class
(*passed* / *failed* / *could-not-run*). Everything in §2 exists to hold that.

---

## 2. The guardrail cross-repo contract

### 2.1 Mirror direction is per file, and it is not always the same way

Check the module docstring before assuming. Getting this backwards silently
reverses which side a fix belongs on.

| File | Canonical | Mirrored into |
|---|---|---|
| `guardrail/topics.py` | **plane** | SDK |
| `guardrail/hazard_taxonomy.py` | **plane** | SDK |
| `guardrail/grounding.py` | **plane** | SDK |
| `_internal/safety_detectors.py` | **SDK** | plane (`agents/services/detectors.py`) |

A type's behaviour spans **two** modules, and pinning one is not enough:

```
config → [resolver] → entities/backend → [detector] → matches → [summary] → row
```

The **config resolver** and the **result shape** are the plane's — that is where
a rule is authored, validated on write and stored durably. The **detectors** are
ours. `pii` shipped faithfully on both sides in 1.60.0 and still diverged four
ways, because only the detector half was mirrored and tested.

### 2.2 The wire rule — settled twice, do not re-litigate

- **A new `implementation_type` *value* is not a wire event.** The plane ships the
  column with no allow-list and `from_policy` has skipped types it cannot rebuild
  since before v1.9.
- **A new *key* on a payload is.** Wire v1.9 was bought by
  `action`/`severity`/`floor`.

Recorded in the plane's `docs/compatibility-matrix.md`. If you find yourself about
to bump the wire for a new type, stop.

### 2.3 The conformance fixture

`tests/data/guardrail_conformance.json` is byte-identical to the plane's
`backend/app/data/guardrail_conformance.json`, and both repos run it through a
thin adapter. The protocol (plane `docs/Guardrail_Type_Contract.md` §2):

1. **Add cases before changing behaviour.**
2. **Never edit one copy.**
3. On a mismatch, decide which side owns the behaviour and change **both**.

It currently covers `pii`, `secrets` and masking. Its `mask` section has **no
`raises` case**, which is why one resolver divergence survived the release that
claimed to fix it — if you touch masking, add one.

### 2.4 The invariant this project keeps breaking

> **A guardrail whose configuration cannot check anything must report that it
> could not run — never a clean verdict.**

Shipped violations in three consecutive releases: `topic`'s `return []` (1.58.0),
`schema`'s empty schema (1.59.0), `pii`'s empty entity list (1.61.0). Each fix
closed one instance.

`tests/test_guardrail_unusable_config_sweep.py` is the sweep that closes the
*class*. It fails when a new distributable type is added without an
unusable-config case. Its `xfail`s are known, signed-off-pending gaps — treat a
new one as a decision, not a formality.

### 2.5 Egress

Local capture is always full fidelity; filtering happens **on the way out**. A
span has **three** content channels and all three are gated:

| Channel | Registry | Filter |
|---|---|---|
| `attributes` | `SENSITIVE_ATTR_KEYS` | `apply_export_policy` |
| `events` | `SENSITIVE_EVENT_ATTR_KEYS` | `apply_event_export_policy` |
| `status.description` | — | `otel._filtered_status` |

Adding a payload-bearing attribute anywhere means a one-line change to the
registry, or it egresses even when the operator set
`FASTAIAGENT_TRACE_PAYLOADS=0`. Guardrail metadata additionally passes through
`guardrail.executor.EXPORTABLE_DETAIL_KEYS`, which is an **allowlist, not a
filter**: a type absent from it exports nothing.

---

## 3. Conventions

- **Never push to `main`.** Branch + PR, including for releases and version bumps.
- **Version sync in one commit:** `pyproject.toml`, `fastaiagent/_version.py`, the
  README PyPI badge, and `CHANGELOG.md`. A fix release still takes a minor bump
  here — see 1.59.0, 1.61.0, 1.62.0.
- **Docs are part of the change**, not a follow-up.
- **Run it before claiming it:** `pytest`, `ruff check`, `ruff format --check`,
  `mypy`, `mkdocs build --strict`. Report actual output.
- **Read the skip lines, not the pass count.** Several suites skip silently
  without Postgres, `FAENT_SDK_PATH`, `presidio`, or a model key. *A skip is not a
  check* — that is how `pii` diverged four ways with a green board.
- **No mocking unless agreed.** Tests exercise the real library. Where a test
  needs a model, mark it `e2e` and put it in `tests/e2e/`; keys live in `~/.zshrc`,
  so wrap live runs in `zsh -lc '…'`.
- **A test that greps source certifies rather than checks.** Both repos have been
  burned by this — a call-site test once passed while four sites leaked, because
  they imported the function under an alias. Assert on behaviour.
- **Clean core.** The default dependency tree must stay AGPL/GPL-free; the
  `clean-core` CI job enforces it. Anything heavy or copyleft goes in an extra —
  except where an extra would make a *safety control* silently under-enforce, which
  is why `jsonschema` is core.
- Planning and analysis docs go in `claude_files/` (gitignored).
