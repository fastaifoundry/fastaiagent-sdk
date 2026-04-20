import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ChevronLeft, Download, Play, RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { SpanInspector } from "@/components/traces/SpanInspector";
import { SpanTree } from "@/components/traces/SpanTree";
import { TraceSummaryBar } from "@/components/traces/TraceSummaryBar";
import { useTrace, useTraceSpans } from "@/hooks/use-traces";
import type { SpanRow } from "@/lib/types";

export function TraceDetailPage() {
  const { traceId } = useParams<{ traceId: string }>();
  const trace = useTrace(traceId);
  const spans = useTraceSpans(traceId);
  const [selectedSpanId, setSelectedSpanId] = useState<string | null>(null);

  const selectedSpan: SpanRow | null = useMemo(() => {
    if (!trace.data?.spans) return null;
    if (!selectedSpanId) return trace.data.spans[0] ?? null;
    return trace.data.spans.find((s) => s.span_id === selectedSpanId) ?? null;
  }, [trace.data, selectedSpanId]);

  const durationMs = useMemo(() => {
    if (!trace.data) return null;
    const start = new Date(trace.data.start_time).getTime();
    const end = new Date(trace.data.end_time).getTime();
    if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
    return Math.max(0, end - start);
  }, [trace.data]);

  if (trace.isLoading || spans.isLoading) {
    return (
      <div className="space-y-4">
        <TableSkeleton rows={6} />
      </div>
    );
  }

  if (trace.error || !trace.data) {
    return (
      <EmptyState
        title="Trace not found"
        description="This trace doesn't exist in the local database. It may have been cleared or never ran."
      />
    );
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title={trace.data.name || "Trace"}
        description={`Detailed view of ${trace.data.span_count} span${
          trace.data.span_count === 1 ? "" : "s"
        }`}
      >
        <Link to="/traces">
          <Button variant="ghost" size="sm">
            <ChevronLeft className="mr-1.5 h-3.5 w-3.5" />
            Back
          </Button>
        </Link>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            trace.refetch();
            spans.refetch();
          }}
          disabled={trace.isFetching || spans.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${
              trace.isFetching || spans.isFetching ? "animate-spin" : ""
            }`}
          />
          Refresh
        </Button>
        <a
          href={`/api/traces/${trace.data.trace_id}/export`}
          target="_blank"
          rel="noreferrer"
        >
          <Button variant="outline" size="sm">
            <Download className="mr-1.5 h-3.5 w-3.5" />
            Export
          </Button>
        </a>
        <Link to={`/traces/${trace.data.trace_id}/replay`}>
          <Button size="sm">
            <Play className="mr-1.5 h-3.5 w-3.5" />
            Open in Replay
          </Button>
        </Link>
      </PageHeader>

      <TraceSummaryBar trace={trace.data} duration_ms={durationMs} />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1.2fr_1fr]">
        <div className="min-w-0">
          {spans.data?.tree ? (
            <SpanTree
              tree={spans.data.tree}
              selectedSpanId={selectedSpan?.span_id ?? null}
              onSelect={(s) => setSelectedSpanId(s.span_id)}
            />
          ) : (
            <EmptyState title="No spans" description="This trace has no spans." />
          )}
        </div>
        <div className="min-w-0">
          <SpanInspector span={selectedSpan} />
        </div>
      </div>
    </div>
  );
}
