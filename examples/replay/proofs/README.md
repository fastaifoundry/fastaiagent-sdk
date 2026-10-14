# Proofs behind "Where a replay has to hold"

The claims on [Where a replay has to hold](../../../docs/replay/replay-boundaries.md),
each as a script you can run against the published SDK. No API key, no network:
the agents are the SDK's own `FunctionModel` and the store is a throwaway `local.db`.
Recorded reruns need no provider, which is the point.

| Section of the page | Script | What it shows |
|---|---|---|
| 1 · Between the trace and the agent | `proof_blueprint.py` | what the root span carries, what `Agent.from_dict` rebuilds from it, and what it leaves behind |
| 2 · Between the model and the record | `proof_recorded.py` | a recorded rerun serves both captured turns in order, calls no model, and is byte-identical; a prompt change can't change a recorded answer |
| 3 · Between the record and the world | `proof_tools.py` | a rerun blocks every tool by default; `with_tool_policy("live", allow=[...])` runs the real function again in the defining process and finds none in a fresh one; an override replaces it; `replay_class` is never inferred |
| 4 · Between the recording and the rerun | `proof_miss.py` | a rerun that needs more model turns than were captured stops with `on_miss="error"`, or warns and goes live |
| 5 · Between a rerun and production | `proof_isolation.py` | a rerun has no memory, re-runs its guardrails, and cannot hold a pause |
| 6 · Between a fix and the future | `proof_regression.py` | `save_as_test` writes the case `evaluate()` reads, with the trail back to the trace |
| 7 · Between a rerun and the record of production | `proof_provenance.py` | a rerun's trace is marked `fastaiagent.source=replay`, is never exported, is left out of production readers, and can go to a store of its own |

```sh
cd examples/replay/proofs
for p in blueprint recorded tools miss isolation regression provenance; do python proof_$p.py || break; done
```

`../tests/test_proofs.py` runs these seven scripts in CI, so the outputs the page
quotes can't drift from what the SDK does. The real-model runs on the page use the
same flow as [`examples/04_agent_replay.py`](../../04_agent_replay.py) and
[`examples/105_decision_replay.py`](../../105_decision_replay.py).
