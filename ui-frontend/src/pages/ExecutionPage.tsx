import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, ChevronRight, RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { JsonViewer } from "@/components/shared/JsonViewer";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/shared/EmptyState";
import {
  useExecution,
  type ExecutionCheckpoint,
} from "@/hooks/use-execution";

function StatusBadge({ status }: { status: string }) {
  const tone =
    status === "interrupted"
      ? "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400"
      : status === "failed"
        ? "bg-red-500/10 text-red-700 dark:text-red-400"
        : "bg-green-500/10 text-green-700 dark:text-green-400";
  return (
    <span className={`rounded px-2 py-0.5 text-xs font-medium ${tone}`}>
      {status}
    </span>
  );
}

function shortChain(path: string | null): string {
  return path || "—";
}

export function ExecutionPage() {
  const { execution_id } = useParams<{ execution_id: string }>();
  const { data, isLoading, isFetching, refetch, error } = useExecution(execution_id);
  const [selected, setSelected] = useState<ExecutionCheckpoint | null>(null);

  if (!execution_id) return null;

  const checkpoints = data?.checkpoints ?? [];
  const latest = checkpoints[checkpoints.length - 1];

  return (
    <div className="space-y-5">
      <PageHeader
        title="Execution"
        description={
          data
            ? `${data.chain_name} · ${data.checkpoint_count} checkpoints · latest: ${data.status}`
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
          {/* Resume CTA only when the latest checkpoint is interrupted/failed
              — recoverable runs. Approve flow stays in /approvals/:id. */}
          {latest && (latest.status === "interrupted" || latest.status === "failed") && (
            <Card className="border-yellow-500/40 bg-yellow-500/5">
              <CardContent className="flex items-center justify-between gap-3 py-4">
                <div className="text-sm">
                  <p className="font-medium">
                    This execution is recoverable.
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Latest checkpoint status:{" "}
                    <StatusBadge status={latest.status} />
                  </p>
                </div>
                <div className="flex gap-2">
                  {latest.status === "interrupted" && (
                    <Button asChild variant="outline" size="sm">
                      <Link to={`/approvals/${execution_id}`}>
                        Open approval
                      </Link>
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle className="text-base">
                  Checkpoints (chronological)
                </CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead className="bg-muted/40 text-xs font-mono uppercase tracking-widest text-muted-foreground">
                      <tr>
                        <th className="py-2 px-3 w-10">#</th>
                        <th className="py-2 px-3">node_id</th>
                        <th className="py-2 px-3">status</th>
                        <th className="py-2 px-3">agent_path</th>
                        <th className="py-2 px-3">created_at</th>
                        <th className="py-2 px-3 w-8"></th>
                      </tr>
                    </thead>
                    <tbody>
                      {checkpoints.map((cp, i) => (
                        <tr
                          key={cp.checkpoint_id}
                          onClick={() => setSelected(cp)}
                          className={`border-b border-border cursor-pointer hover:bg-muted/30 ${
                            selected?.checkpoint_id === cp.checkpoint_id
                              ? "bg-muted/40"
                              : ""
                          }`}
                        >
                          <td className="py-2 px-3 text-xs text-muted-foreground">
                            {i + 1}
                          </td>
                          <td className="py-2 px-3 font-mono text-xs">
                            {cp.node_id}
                          </td>
                          <td className="py-2 px-3">
                            <StatusBadge status={cp.status} />
                          </td>
                          <td className="py-2 px-3 font-mono text-[11px] text-muted-foreground">
                            {shortChain(cp.agent_path)}
                          </td>
                          <td className="py-2 px-3 text-xs text-muted-foreground">
                            {cp.created_at}
                          </td>
                          <td className="py-2 px-3">
                            <ChevronRight className="h-4 w-4 text-muted-foreground" />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-base">
                  state_snapshot
                </CardTitle>
              </CardHeader>
              <CardContent>
                {selected ? (
                  <>
                    <p className="mb-2 text-xs text-muted-foreground">
                      Click a different checkpoint on the left to inspect its
                      state.
                    </p>
                    <JsonViewer
                      data={selected.state_snapshot}
                      className="max-h-[60vh] p-3"
                    />
                  </>
                ) : (
                  <p className="text-xs text-muted-foreground">
                    Click a checkpoint to inspect its state_snapshot.
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
