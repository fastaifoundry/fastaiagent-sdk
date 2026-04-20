import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ChevronLeft, GitFork, RefreshCw, Save } from "lucide-react";
import { toast } from "sonner";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/shared/EmptyState";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { SpanInspector } from "@/components/traces/SpanInspector";
import { SpanTree } from "@/components/traces/SpanTree";
import { TraceSummaryBar } from "@/components/traces/TraceSummaryBar";
import { ReplayDiffView } from "@/components/replay/ReplayDiffView";
import { ReplayForkDialog } from "@/components/replay/ReplayForkDialog";
import { SaveAsTestDialog } from "@/components/replay/SaveAsTestDialog";
import { useTrace, useTraceSpans } from "@/hooks/use-traces";
import { useCompareFork, useReplay } from "@/hooks/use-replay";
import type { ComparisonResult, RerunResult, ReplayStep, SpanRow } from "@/lib/types";
import { ApiError } from "@/lib/api";

function stringify(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function AgentReplayPage() {
  const { traceId } = useParams<{ traceId: string }>();
  const trace = useTrace(traceId);
  const spans = useTraceSpans(traceId);
  const replay = useReplay(traceId);
  const compareFork = useCompareFork();

  const [selectedSpanId, setSelectedSpanId] = useState<string | null>(null);
  const [forkDialogOpen, setForkDialogOpen] = useState(false);
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [rerun, setRerun] = useState<RerunResult | null>(null);
  const [comparison, setComparison] = useState<ComparisonResult | null>(null);

  const selectedSpan: SpanRow | null = useMemo(() => {
    if (!trace.data?.spans) return null;
    if (!selectedSpanId) return trace.data.spans[0] ?? null;
    return trace.data.spans.find((s) => s.span_id === selectedSpanId) ?? null;
  }, [trace.data, selectedSpanId]);

  const selectedStep: ReplayStep | null = useMemo(() => {
    if (!replay.data?.steps || !selectedSpan) return null;
    return (
      replay.data.steps.find((s) => s.span_id === selectedSpan.span_id) ?? null
    );
  }, [replay.data, selectedSpan]);

  const durationMs = useMemo(() => {
    if (!trace.data) return null;
    const start = new Date(trace.data.start_time).getTime();
    const end = new Date(trace.data.end_time).getTime();
    if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
    return Math.max(0, end - start);
  }, [trace.data]);

  // Once a rerun lands, immediately fetch the step-by-step comparison.
  useEffect(() => {
    if (!rerun || !rerun.new_trace_id) return;
    compareFork
      .mutateAsync({ forkId: rerun.fork_id, against: rerun.new_trace_id })
      .then(setComparison)
      .catch((e) => {
        if (e instanceof ApiError) toast.error(e.message);
        else toast.error("Comparison failed");
      });
    // compareFork mutation is stable; excluded from deps intentionally.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rerun]);

  if (trace.isLoading || spans.isLoading) {
    return <TableSkeleton rows={6} />;
  }
  if (trace.error || !trace.data) {
    return (
      <EmptyState
        title="Trace not found"
        description="Can't replay a trace that doesn't exist locally."
      />
    );
  }

  const forkDisabled = !selectedStep;

  return (
    <div className="space-y-5">
      <PageHeader
        title="Agent Replay"
        description={`Fork at any span, modify, rerun — then compare side-by-side.`}
      >
        <Link to={`/traces/${trace.data.trace_id}`}>
          <Button variant="ghost" size="sm">
            <ChevronLeft className="mr-1.5 h-3.5 w-3.5" />
            Trace detail
          </Button>
        </Link>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            trace.refetch();
            spans.refetch();
            replay.refetch();
          }}
          disabled={trace.isFetching || spans.isFetching || replay.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${
              trace.isFetching || spans.isFetching || replay.isFetching
                ? "animate-spin"
                : ""
            }`}
          />
          Refresh
        </Button>
        <Button
          size="sm"
          onClick={() => setForkDialogOpen(true)}
          disabled={forkDisabled}
          title={forkDisabled ? "Select a span to fork from" : undefined}
        >
          <GitFork className="mr-1.5 h-3.5 w-3.5" />
          Fork here
        </Button>
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

      {rerun && (
        <div className="space-y-3 rounded-md border-2 border-primary/30 bg-card p-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold">Rerun complete</h2>
              <p className="text-xs text-muted-foreground">
                Forked from step {selectedStep?.step ?? "—"} ·{" "}
                {rerun.new_trace_id ? (
                  <Link
                    to={`/traces/${rerun.new_trace_id}`}
                    className="font-mono text-primary hover:underline"
                  >
                    {rerun.new_trace_id.slice(0, 16)}…
                  </Link>
                ) : (
                  "no trace id returned"
                )}
              </p>
            </div>
            <Button
              size="sm"
              variant="outline"
              onClick={() => setSaveDialogOpen(true)}
            >
              <Save className="mr-1.5 h-3.5 w-3.5" />
              Save as regression test
            </Button>
          </div>

          {comparison ? (
            <ReplayDiffView
              comparison={comparison}
              originalOutput={rerun.original_output}
              newOutput={rerun.new_output}
            />
          ) : compareFork.isPending ? (
            <TableSkeleton rows={4} />
          ) : (
            <p className="text-xs text-muted-foreground">
              Comparison not available.
            </p>
          )}
        </div>
      )}

      <ReplayForkDialog
        traceId={trace.data.trace_id}
        step={selectedStep}
        open={forkDialogOpen}
        onOpenChange={setForkDialogOpen}
        onRerunComplete={(_, result) => setRerun(result)}
      />

      {rerun && (
        <SaveAsTestDialog
          open={saveDialogOpen}
          onOpenChange={setSaveDialogOpen}
          forkId={rerun.fork_id}
          originalInput={stringify(selectedStep?.input ?? trace.data.name)}
          originalOutput={stringify(rerun.new_output ?? rerun.original_output)}
        />
      )}
    </div>
  );
}
