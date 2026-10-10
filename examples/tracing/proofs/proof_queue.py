"""Proof for "4 · Between your process and the plane" on docs/tracing/trace-boundaries.md.

Offline: the SDK's own FunctionModel, no API key, a throwaway local.db, and a
plane address that nothing listens on.

Spans land in local.db marked unsent. The plane exporter drains unsent rows from
the table, not from the batch OTel hands it, and marks them sent only after the
plane acknowledged them. With no plane, the rows stay unsent and the run is
unaffected.
"""

import os
import tempfile
import time

os.environ["FASTAIAGENT_LOCAL_DB"] = os.path.join(tempfile.mkdtemp(), "local.db")

import fastaiagent as fa  # noqa: E402
from fastaiagent.testing import FunctionModel  # noqa: E402
from fastaiagent.trace import TraceStore  # noqa: E402
from fastaiagent.trace.otel import get_tracer_provider  # noqa: E402

t0 = time.perf_counter()
fa.connect(api_key="fa_k_offline-proof", target="http://127.0.0.1:9")  # nothing listens here
print(
    f"connect() to a dead plane returned in {time.perf_counter() - t0:.2f}s; "
    f"is_connected={fa.is_connected}"
)

store = TraceStore()
agent = fa.Agent(name="support", llm=FunctionModel(lambda m: "Order 1042 has shipped."))
t0 = time.perf_counter()
for _ in range(3):
    agent.run("Where is order 1042?")
print(f"3 runs took {time.perf_counter() - t0:.2f}s with the plane down")
get_tracer_provider().force_flush()

unsent = store.count_unsynced()
rows = len(store.list_spans(since=0, limit=100))
print(f"rows in local.db: {rows}   unsent: {unsent}")
pending = store.fetch_unsynced(limit=10)
print(
    f"fetch_unsynced() hands the exporter {len(pending)} spans: {sorted({s.name for s in pending})}"
)

# What the exporter does after a 2xx: mark exactly those spans sent.
store.mark_synced([s.span_id for s in pending[:2]])
print(f"after marking 2 sent: unsent={store.count_unsynced()}  rows={rows}")
fa.disconnect()
