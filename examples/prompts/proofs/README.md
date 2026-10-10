# Proofs behind "A Prompt Breaks at the Boundaries"

The six claims on [A Prompt Breaks at the Boundaries](../../../docs/prompts/prompt-boundaries.md),
each as a script you can run against the published SDK. Every script works in a
scratch folder of its own — a fresh `local.db`, no control plane — so nothing
touches your project's registry.

| Section of the page | Script | Needs | What it shows |
|---|---|---|---|
| 1 · Between the template and the text | `proof_1_placeholders.py` | nothing | fragments resolve at load, variables at format; an unknown fragment and an unfilled variable stay in the text |
| 2 · Between one version and the next | `proof_2_versions.py` | nothing | versions accumulate, the alias stays put; `register(version=1)` replaces row v1 in place |
| 3 · Between a fragment and the prompts that use it | `proof_3_fragments.py` | nothing | editing a fragment changes every prompt that uses it at the same version; the trace holds the text sent |
| 4 · Between the prompt and the run | `proof_4_lineage.py` | nothing | the `Prompt` object stamps every llm span on every path; a formatted string stamps nothing; a nested agent stamps its own |
| 5 · Between the prompt and the model | `proof_5_eval_gate.py` | `OPENAI_API_KEY` | two versions on ten tickets with `gpt-4.1-mini`; the alias moves when the number says so |
| 6 · Between your laptop and the plane | `proof_6_plane_side.py` | nothing | two folders are two registries; what a pushed agent sends; the plane paths refuse offline |

```sh
cd examples/prompts/proofs
python proof_1_placeholders.py && python proof_2_versions.py && python proof_3_fragments.py \
  && python proof_4_lineage.py && python proof_6_plane_side.py
zsh -lc 'python proof_5_eval_gate.py'     # the live one
```

`../tests/test_proofs.py` runs the five offline scripts in CI, so the outputs the page
quotes can't drift from what the SDK does.
