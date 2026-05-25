import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ChevronLeft, Play, RefreshCw, EyeOff } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { SpanInspector } from "@/components/traces/SpanInspector";
import { SpanTree } from "@/components/traces/SpanTree";
import { TraceComparePicker } from "@/components/traces/TraceComparePicker";
import { TraceScoresCard } from "@/components/traces/TraceScoresCard";
import { TraceSummaryBar } from "@/components/traces/TraceSummaryBar";
import { ExportTraceDialog } from "@/components/traces/ExportTraceDialog";
import { useTrace, useTraceSpans } from "@/hooks/use-traces";
import type { SpanRow } from "@/lib/types";

export function TraceDetailPage() {
  const { traceId } = useParams<{ traceId: string }>();
  // ``redact`` flips ``?redact=true`` on the trace API calls. The backend
  // only honors it when a ``RedactionPolicy(mode in {"read", "both"})``
  // is installed via ``fastaiagent.trace.set_redaction_policy(...)``;
  // otherwise the flag is a no-op. See docs/security.md.
  const [redact, setRedact] = useState(false);
  const trace = useTrace(traceId, redact);
  const spans = useTraceSpans(traceId, redact);
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
        <label
          className="ml-1 flex items-center gap-2 rounded-md border px-2.5 py-1 text-xs font-mono"
          title={
            "Sends ?redact=true to the trace API. Honored only when a " +
            "RedactionPolicy(mode in {read, both}) is installed via " +
            "fastaiagent.trace.set_redaction_policy(...). See docs/security.md."
          }
        >
          <EyeOff className="h-3.5 w-3.5 text-muted-foreground" />
          <span>Mask secrets</span>
          <Switch
            checked={redact}
            onCheckedChange={setRedact}
            aria-label="Toggle trace attribute redaction"
            size="sm"
          />
        </label>
        <ExportTraceDialog traceId={trace.data.trace_id} />
        <TraceComparePicker pinnedTraceId={trace.data.trace_id} />
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

      <TraceScoresCard traceId={trace.data.trace_id} />
    </div>
  );
}
