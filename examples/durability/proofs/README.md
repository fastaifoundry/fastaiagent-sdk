# Proofs behind "A Durable Run Breaks at the Boundaries"

The six claims on [A Durable Run Breaks at the Boundaries](../../../docs/durability/durability-boundaries.md),
each as a script you can run against the published SDK. Every script works on a
scratch SQLite file of its own, with a real agent on the SDK's offline model and real
child processes that are killed where it matters. No API key.

| Section of the page | Script | Needs | What it shows |
|---|---|---|---|
| 1 · Between one step and the next | `proof_1_rows.py` | nothing | the rows a completed run and a paused run write; the pause row and pending row; no marker on a pause |
| 2 · Between the process that paused and the one that resumes | `proof_2_pause_resume.py` | nothing | pause in a child, resume in the parent; the frozen context; `AlreadyResumed`; five racers, one winner |
| 3 · Between a crash and a rerun | `proof_3_crash.py` | nothing | killed in the tool, killed in the model, a sibling tool call; the crash-then-pause edge |
| 4 · Between a resume and a side effect | `proof_4_side_effects.py` | nothing | a charge fires twice plain, once under `@idempotent`; the cache is per run |
| 5 · Between a run that ended and one that died | `proof_5_run_end.py` | nothing | the run-end marker: resume refused, failed stays resumable, fork, prune keeps a pause |
| 6 · Between a laptop and a fleet, and the plane | `proof_6_backends.py` | `PG_TEST_DSN` for the Postgres half | the race on SQLite and Postgres; the outbox while disconnected |

```sh
cd examples/durability/proofs
python proof_1_rows.py && python proof_2_pause_resume.py && python proof_3_crash.py \
  && python proof_4_side_effects.py && python proof_5_run_end.py
scripts/dev_backends.sh up    # from the repo root, for the Postgres half
PG_TEST_DSN="postgresql://postgres:test@127.0.0.1:55432/fastaiagent_test" python proof_6_backends.py
```

`_agents.py` is the refund bot every proof runs: its model decides from the conversation
rather than a counter, which is what lets a run paused in one process be resumed in another.

`../tests/test_proofs.py` runs all six in CI (the Postgres half skips without a DSN), so
the outputs the page quotes can't drift from what the SDK does.
