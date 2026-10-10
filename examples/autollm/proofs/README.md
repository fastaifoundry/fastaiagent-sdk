# Proofs behind "How AutoLLM works"

The four claims on [How AutoLLM works](../../../docs/evaluation/autollm-how-it-works.md)
that don't need a live model, each as a script you can run against the published
SDK. No API key, no network: the agents are the SDK's own `FunctionModel`.

| Section of the page | Script | What it shows |
|---|---|---|
| 3 · One candidate, one agent | `proof_isolation.py` | a tuned candidate leaves the original untouched and carries nothing between two customers |
| 4 · A crash is not a pass | `proof_errors.py` | a candidate that raises on its hard cases scores 0.5, not the 1.0 `evaluate()` alone would report |
| 5 · The judge that picks doesn't grade | `proof_judges.py` | two judges with one name are refused up front; an unset audit judge warns |
| 6 · The bill has a ceiling | `proof_budget.py` | a cap too small is refused before a call; a cap of 5 writes exactly 5 evaluation passes |

```sh
cd examples/autollm/proofs
python proof_isolation.py && python proof_errors.py && python proof_judges.py && python proof_budget.py
```

Sections 1 and 2 are proved by real runs of the live examples instead —
[`../../autollm-loop/`](../../autollm-loop/) and [`../calibrate_judge.py`](../calibrate_judge.py).
`tests/test_proofs.py` runs these four scripts in CI, so the outputs the page quotes
can't drift from what the SDK does.
