# Proofs behind "A Guardrail Breaks at the Boundaries"

The six claims on [A Guardrail Breaks at the Boundaries](../../../docs/guardrails/guardrail-boundaries.md),
each as a script you can run against the published SDK. Every script works in a
scratch folder of its own, with real agents on the SDK's offline models, real rules
and real spans.

| Section of the page | Script | Needs | What it shows |
|---|---|---|---|
| 1 · Between the gate and the run | `proof_1_positions.py` | nothing | blocking rules first and fail-fast, observers after; an observer's crash is a recorded failure; four positions on one tool call |
| 2 · Between a failure and what it costs | `proof_2_actions.py` | nothing | block, warn, mask, override, reask; a mask feeds the next rule and the model; a streamed reply blocks |
| 3 · "Could not run" vs "found nothing" | `proof_3_could_not_run.py` | nothing | `on_error`; ten unusable configurations all report `errored`; errored always blocks |
| 3 · the model-backed judges | `proof_3b_judges.py` | `OPENAI_API_KEY` | a `topic` rule names what it found; `groundedness` errors without its context slot |
| 4 · Between the verdict and the trace | `proof_4_trace.py` | nothing | one span per rule; the detail allowlist; what the export policy drops |
| 5 · Between your code and the plane's rule | `proof_5_plane_rule.py` | nothing | a plane rule rebuilt onto the SDK's runners; the shared conformance fixture, 26/26 |
| 6 · Beyond the agent loop | `proof_6_roundtrip.py` | nothing | what survives `to_dict`/`from_dict`; what a foreign-framework proxy can honour |

```sh
cd examples/guardrails/proofs
python proof_1_positions.py && python proof_2_actions.py && python proof_3_could_not_run.py \
  && python proof_4_trace.py && python proof_5_plane_rule.py && python proof_6_roundtrip.py
zsh -lc 'python proof_3b_judges.py'       # the live one
```

`../tests/test_proofs.py` runs the six offline scripts in CI, so the outputs the page
quotes can't drift from what the SDK does.
