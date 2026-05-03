import { Link, useSearchParams } from "react-router-dom";
import { ArrowLeftRight, ChevronLeft } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/shared/EmptyState";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { SpanAlignmentTable } from "@/components/traces/SpanAlignmentTable";
import { TraceCompareSummary } from "@/components/traces/TraceCompareSummary";
import { useCompareTraces } from "@/hooks/use-traces";
import { formatTimeAgo, shortTraceId } from "@/lib/format";
import type { CompareTraceHalf } from "@/lib/types";

function timeApartLabel(seconds: number | null): string {
  if (seconds == null) return "";
  if (seconds < 60) return `${Math.round(seconds)}s apart`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m apart`;
  if (seconds < 86_400) return `${Math.round(seconds / 3600)}h apart`;
  return `${Math.round(seconds / 86_400)}d apart`;
}

function HalfLabel({
  half,
  side,
}: {
  half: CompareTraceHalf;
  side: "A" | "B";
}) {
  return (
    <div className="min-w-0 flex-1">
      <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        Trace {side}
      </div>
      <Link
        to={`/traces/${half.trace_id}`}
        className="truncate font-medium hover:text-primary"
        title={half.trace_id}
      >
        {half.name || shortTraceId(half.trace_id)}
      </Link>
      <div className="text-[11px] font-mono text-muted-foreground">
        {shortTraceId(half.trace_id)} · {formatTimeAgo(half.start_time)}
        {half.agent_name ? ` · ${half.agent_name}` : ""}
      </div>
    </div>
  );
}

/**
 * Side-by-side trace comparison. Generalises Replay's "original vs forked"
 * diff to any two traces — useful for regression detection, A/B prompt
 * testing, and "why is Monday different from Friday" debugging.
 *
 * URL is the source of truth: `/traces/compare?a=<id>&b=<id>` — bookmark
 * or share to deep-link directly to a specific comparison. The "Swap"
 * button just toggles the two query params.
 */
export function TraceComparePage() {
  const [params, setParams] = useSearchParams();
  const a = params.get("a");
  const b = params.get("b");

  const { data, isLoading, error } = useCompareTraces(a, b);

  if (!a || !b) {
    return (
      <div className="space-y-5">
        <PageHeader
          title="Compare traces"
          description="Pick two traces to view their differences side-by-side."
        />
        <EmptyState
          title="Two trace IDs required"
          description="Open a trace and click 'Compare with…', or select two rows in the Traces list."
        />
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="space-y-4">
        <TableSkeleton rows={6} />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="space-y-5">
        <PageHeader title="Compare traces" />
        <EmptyState
          title="Comparison unavailable"
          description={
            error instanceof Error
              ? error.message
              : "One of these traces doesn't exist in this project."
          }
        />
      </div>
    );
  }

  const { trace_a, trace_b, alignment, summary } = data;
  const swap = () => {
    const next = new URLSearchParams(params);
    next.set("a", b);
    next.set("b", a);
    setParams(next, { replace: true });
  };

  return (
    <div className="space-y-5">
      <PageHeader
        title="Compare traces"
        description={timeApartLabel(summary.time_apart_seconds)}
      >
        <Link to="/traces">
          <Button variant="ghost" size="sm">
            <ChevronLeft className="mr-1.5 h-3.5 w-3.5" />
            Back
          </Button>
        </Link>
      </PageHeader>

      <div className="flex items-center gap-3 rounded-md border bg-card px-4 py-3">
        <HalfLabel half={trace_a} side="A" />
        <Button
          variant="outline"
          size="sm"
          onClick={swap}
          aria-label="Swap A and B"
          title="Swap A and B"
        >
          <ArrowLeftRight className="h-3.5 w-3.5" />
        </Button>
        <HalfLabel half={trace_b} side="B" />
      </div>

      <TraceCompareSummary summary={summary} />

      <SpanAlignmentTable
        rows={alignment}
        traceA={trace_a}
        traceB={trace_b}
      />
    </div>
  );
}
