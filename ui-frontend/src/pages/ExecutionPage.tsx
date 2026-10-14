import { Link, useLocation, useParams } from "react-router-dom";
import { ArrowLeft, ExternalLink, RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { EmptyState } from "@/components/shared/EmptyState";
import type { ResumedState } from "@/components/approvals/ApprovalActions";
import { CheckpointTimeline } from "@/components/executions/CheckpointTimeline";
import { ExecutionOutcomeCard } from "@/components/executions/ExecutionOutcomeCard";
import { IdempotencyCacheCard } from "@/components/executions/IdempotencyCacheCard";
import { useExecution } from "@/hooks/use-execution";

export function ExecutionPage() {
  const { execution_id } = useParams<{ execution_id: string }>();
  const { data, isLoading, isFetching, refetch, error } = useExecution(execution_id);
  // Set by ApprovalActions after a resume: the resumed run's trace.
  const resumed = (useLocation().state ?? null) as ResumedState | null;

  if (!execution_id) return null;

  const checkpoints = data?.checkpoints ?? [];
  const latest = checkpoints[checkpoints.length - 1];
  // 1.89.0: the server says whether the pause still waits. An older server
  // doesn't (undefined), and then the latest checkpoint is all we have.
  const stillWaiting =
    data?.pending_interrupt !== undefined
      ? data.pending_interrupt !== null
      : latest?.status === "interrupted";

  return (
    <div className="space-y-5">
      <PageHeader
        title="Execution"
        description={
          data
            ? `${data.chain_name} · ${data.checkpoint_count} checkpoint${data.checkpoint_count === 1 ? "" : "s"} · latest: ${data.status}`
            : isLoading
              ? "Loading…"
              : "Execution not found."
        }
      >
        <div className="flex items-center gap-2">
          <Link
            to="/approvals"
            className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="h-4 w-4" />
            Approvals
          </Link>
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            disabled={isFetching}
          >
            <RefreshCw
              className={`mr-1.5 h-3.5 w-3.5 ${isFetching ? "animate-spin" : ""}`}
            />
            Refresh
          </Button>
        </div>
      </PageHeader>

      {error && (
        <Card className="border-destructive/50">
          <CardContent className="pt-4 text-sm text-destructive">
            {error instanceof Error ? error.message : String(error)}
          </CardContent>
        </Card>
      )}

      {data && checkpoints.length === 0 && (
        <EmptyState
          title="No checkpoints"
          description="This execution has no checkpoints recorded."
        />
      )}

      {data && (data.quarantined_count ?? 0) > 0 ? (
        <Card
          className="border-destructive/40 bg-destructive/5"
          data-testid="quarantine-banner"
        >
          <CardContent className="py-4 text-sm">
            <p className="font-medium">
              The platform&rsquo;s copy of this run is incomplete.
            </p>
            <p className="text-xs text-muted-foreground">
              {data.quarantined_count} checkpoint
              {data.quarantined_count === 1 ? "" : "s"} could not be replicated
              and {data.quarantined_count === 1 ? "was" : "were"} quarantined so
              the outbox could keep moving. Local durability is unaffected —
              resume still works from this machine.
            </p>
          </CardContent>
        </Card>
      ) : null}

      {resumed?.resumedTraceId && (
        <Card className="border-fa-success/40 bg-fa-success/5" data-testid="resumed-banner">
          <CardContent className="flex items-center justify-between gap-3 py-3 text-sm">
            <span>
              Resumed{resumed.approved === false ? " (rejected)" : resumed.approved ? " (approved)" : ""}.
            </span>
            <Link
              to={`/traces/${resumed.resumedTraceId}`}
              className="inline-flex items-center gap-1 text-primary hover:underline"
            >
              <ExternalLink className="h-3.5 w-3.5" />
              View trace
            </Link>
          </CardContent>
        </Card>
      )}

      {data && checkpoints.length > 0 && (
        <>
          <ExecutionOutcomeCard detail={data} />

          {latest &&
          ((latest.status === "interrupted" && stillWaiting) ||
            latest.status === "failed") ? (
            <Card className="border-amber-500/40 bg-amber-500/5">
              <CardContent className="flex items-center justify-between gap-3 py-4">
                <div className="text-sm">
                  <p className="font-medium">This execution is recoverable.</p>
                  <p className="text-xs text-muted-foreground">
                    Latest checkpoint status:{" "}
                    <span className="font-mono">{latest.status}</span>
                    {latest.interrupt_reason
                      ? ` · ${latest.interrupt_reason}`
                      : ""}
                  </p>
                </div>
                <div className="flex gap-2">
                  {latest.status === "interrupted" ? (
                    <Button asChild variant="outline" size="sm">
                      <Link to={`/approvals/${execution_id}`}>
                        Open approval
                      </Link>
                    </Button>
                  ) : null}
                </div>
              </CardContent>
            </Card>
          ) : null}

          <CheckpointTimeline checkpoints={checkpoints} resolution={data.resolution} />
          <IdempotencyCacheCard executionId={execution_id} />
        </>
      )}
    </div>
  );
}
