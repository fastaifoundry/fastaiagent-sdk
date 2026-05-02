import { Link, useParams } from "react-router-dom";
import { ArrowLeft, RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { EmptyState } from "@/components/shared/EmptyState";
import { CheckpointTimeline } from "@/components/executions/CheckpointTimeline";
import { IdempotencyCacheCard } from "@/components/executions/IdempotencyCacheCard";
import { useExecution } from "@/hooks/use-execution";

export function ExecutionPage() {
  const { execution_id } = useParams<{ execution_id: string }>();
  const { data, isLoading, isFetching, refetch, error } = useExecution(execution_id);

  if (!execution_id) return null;

  const checkpoints = data?.checkpoints ?? [];
  const latest = checkpoints[checkpoints.length - 1];

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

      {data && checkpoints.length > 0 && (
        <>
          {latest &&
          (latest.status === "interrupted" || latest.status === "failed") ? (
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

          <CheckpointTimeline checkpoints={checkpoints} />
          <IdempotencyCacheCard executionId={execution_id} />
        </>
      )}
    </div>
  );
}
