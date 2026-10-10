# Proofs behind "Where a trace has to hold"

The claims on [Where a trace has to hold](../../../docs/tracing/trace-boundaries.md),
each as a script you can run against the published SDK. No API key, no network:
the agents are the SDK's own `FunctionModel`, the store is a throwaway `local.db`,
and the exporter is OpenTelemetry's own `InMemorySpanExporter`.

| Section of the page | Script | What it shows |
|---|---|---|
| 1 · Between calls | `proof_tree.py` | two agents run at once; each trace is flat rows with one root, rebuilt into its own tree; a span opened inside a tool nests under it |
| 2 · Between the span and the disk | `proof_durable.py` | a span is on disk the moment it ends; a tool error is a result, a model error is an ERROR root; an unwritable store never fails the run |
| 3 · Between capture and egress | `proof_egress.py` | `local.db` keeps everything; `FASTAIAGENT_TRACE_PAYLOADS=0` strips payloads on the way out; a `RedactionPolicy` masks; `FASTAIAGENT_TRACE_ENABLED=0` captures nothing |
| 4 · Between your process and the plane | `proof_queue.py` | with no plane, spans wait in `local.db` marked unsent, and the run is not blocked |
| 5 · Between frameworks | `proof_foreign.py` | an OpenInference span lands in the same table with the canonical keys filled in |

```sh
cd examples/tracing/proofs
python proof_tree.py && python proof_durable.py && python proof_egress.py && python proof_queue.py && python proof_foreign.py
```

`../tests/test_proofs.py` runs these five scripts in CI, so the outputs the page
quotes can't drift from what the SDK does. The real-model trace on the page comes
from the same agent as [`examples/04_agent_replay.py`](../../04_agent_replay.py).
