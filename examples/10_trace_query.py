"""Example 10: Query, search, and export locally stored traces.

Traces are stored automatically in a local SQLite database every time
an agent or chain runs. This example shows how to use TraceStore to
list, inspect, search, and export those traces.

The SQLite database defaults to .fastaiagent/traces.db but can point to
any location — including cloud-mounted filesystems (Azure Files, S3 via
Mountpoint/s3fs, EFS, GCS FUSE, etc.).

Usage:
    python examples/10_trace_query.py

    # Custom storage location (local or mounted):
    FASTAIAGENT_TRACE_DB_PATH=/mnt/azure-share/traces.db python examples/10_trace_query.py
"""

import json

from fastaiagent.trace import TraceStore, trace_context


def create_sample_traces() -> list[str]:
    """Create a few sample traces to query."""
    trace_ids = []

    with trace_context("data-pipeline") as span:
        span.set_attribute("fastai.agent.name", "ingest-bot")
        span.set_attribute("pipeline.stage", "extract")

        with trace_context("fetch-api"):
            pass
        with trace_context("transform-records"):
            pass
        with trace_context("load-warehouse"):
            pass

        trace_ids.append(format(span.get_span_context().trace_id, "032x"))

    with trace_context("support-agent") as span:
        span.set_attribute("fastai.agent.name", "support-bot")
        span.set_attribute("gen_ai.system", "openai")
        span.set_attribute("gen_ai.request.model", "gpt-4.1")

        with trace_context("llm.chat_completion"):
            pass

        trace_ids.append(format(span.get_span_context().trace_id, "032x"))

    return trace_ids


if __name__ == "__main__":
    print("Trace Query Example")
    print("=" * 60)

    # ── 1. Create sample traces ──────────────────────────────────
    trace_ids = create_sample_traces()
    print(f"\nCreated {len(trace_ids)} sample traces.\n")

    # ── 2. Open the trace store ──────────────────────────────────
    # Default path: .fastaiagent/traces.db
    # Override with FASTAIAGENT_TRACE_DB_PATH env var, or pass db_path:
    #   store = TraceStore(db_path="/mnt/azure-share/traces.db")
    #   store = TraceStore(db_path="/mnt/s3-bucket/traces.db")
    store = TraceStore()

    # ── 3. List recent traces ────────────────────────────────────
    print("Recent traces:")
    print("-" * 60)
    traces = store.list_traces(last_hours=24)
    for t in traces:
        print(f"  {t.trace_id[:12]}...  {t.name:<25} spans={t.span_count}  {t.status}")
    print()

    # ── 4. Get a specific trace with all spans ───────────────────
    trace_id = trace_ids[0]
    trace = store.get_trace(trace_id)
    print(f"Trace detail: {trace.name}")
    print(f"  ID:     {trace.trace_id}")
    print(f"  Status: {trace.status}")
    print(f"  Spans:  {len(trace.spans)}")
    for span in trace.spans:
        indent = "    " if span.parent_span_id else "  "
        print(f"{indent}↳ {span.name}  ({span.start_time})")
        if span.attributes:
            for k, v in span.attributes.items():
                print(f"{indent}    {k}={v}")
    print()

    # ── 5. Search traces by name or attribute ────────────────────
    print("Search results for 'support':")
    results = store.search("support")
    for t in results:
        print(f"  {t.trace_id[:12]}...  {t.name}")
    print()

    # ── 6. Export a trace as JSON ────────────────────────────────
    json_str = store.export(trace_id)
    data = json.loads(json_str)
    print(f"Exported trace '{data['name']}' → {len(json_str)} bytes JSON")
    print(f"  Preview: {json_str[:200]}...")
    print()

    # ── 7. Custom storage location examples ──────────────────────
    print("Storage location options:")
    print("  Local (default):  .fastaiagent/traces.db")
    print("  Custom local:     FASTAIAGENT_TRACE_DB_PATH=/data/traces.db")
    print("  Azure Files:      FASTAIAGENT_TRACE_DB_PATH=/mnt/azure-share/traces.db")
    print("  S3 (Mountpoint):  FASTAIAGENT_TRACE_DB_PATH=/mnt/s3-bucket/traces.db")
    print("  EFS:              FASTAIAGENT_TRACE_DB_PATH=/mnt/efs/traces.db")
    print("  GCS FUSE:         FASTAIAGENT_TRACE_DB_PATH=/mnt/gcs-bucket/traces.db")

    store.close()
