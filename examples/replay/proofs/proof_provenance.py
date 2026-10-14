"""Proof for "7 · Between a rerun and the record of production" on docs/replay/replay-boundaries.md.

Offline: the SDK's own FunctionModel, no API key, a throwaway local.db.

Since 1.89.0 a rerun's trace says it is a replay. Every span carries
fastaiagent.source=replay, the root names the trace it replays, and the rows
are written already marked as sent, so no exporter ships them to the plane.
Readers that mean production (list_traces(exclude_sources=True), Analytics,
Overview, Agents, eval curation, memory learning) leave the rerun out. A rerun
can also be written to a store of its own with with_trace_db.
"""

import os
import sqlite3
import tempfile

DIR = tempfile.mkdtemp()
DB = os.path.join(DIR, "local.db")
os.environ["FASTAIAGENT_LOCAL_DB"] = DB

import fastaiagent as fa  # noqa: E402
from fastaiagent.testing import FunctionModel  # noqa: E402
from fastaiagent.trace import Replay, TraceStore  # noqa: E402


def responder(messages):
    return "Order 1042 ships Friday."


agent = fa.Agent(name="shipping", llm=FunctionModel(responder))
r = agent.run("When does order 1042 ship?")
rerun = Replay.load(r.trace_id).fork_at(0).with_determinism("recorded", on_miss="error").rerun()

root = TraceStore().get_trace(rerun.trace_id).spans[0]
print(f"rerun root: fastaiagent.source={root.attributes.get('fastaiagent.source')!r}")
print(f"  replays: {root.attributes.get('fastaiagent.replay.source_trace_id') == r.trace_id}")
print(f"  determinism={root.attributes.get('fastaiagent.replay.determinism')!r}")
print(f"  tool_policy={root.attributes.get('fastaiagent.replay.tool_policy')!r}")
print(f"rerun.exported={rerun.exported}")

with sqlite3.connect(DB) as conn:
    for label, trace_id in (("original", r.trace_id), ("rerun", rerun.trace_id)):
        rows = conn.execute("SELECT synced FROM spans WHERE trace_id = ?", (trace_id,))
        print(f"{label:<8} rows synced={sorted({row[0] for row in rows})}")

all_ids = {t.trace_id for t in TraceStore().list_traces()}
prod_ids = {t.trace_id for t in TraceStore().list_traces(exclude_sources=True)}
print(
    f"list_traces(): {len(all_ids)} traces   exclude_sources=True: {len(prod_ids)} "
    f"(the rerun left out: {rerun.trace_id not in prod_ids})"
)

side = os.path.join(DIR, "experiments.db")
forked = (
    Replay.load(r.trace_id)
    .fork_at(0)
    .with_determinism("recorded", on_miss="error")
    .with_trace_db(side)
)
rerun = forked.rerun()
in_side = rerun.trace_id in {t.trace_id for t in TraceStore(side).list_traces()}
in_local = rerun.trace_id in {t.trace_id for t in TraceStore().list_traces()}
print(f"with_trace_db: rerun in experiments.db: {in_side}   in local.db: {in_local}")
print(f"  compare(): status={forked.compare(rerun).compare_status}")
