import { Link, useParams } from "react-router-dom";
import { ArrowLeft, AlertTriangle } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { JsonViewer } from "@/components/shared/JsonViewer";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ApprovalActions } from "@/components/approvals/ApprovalActions";
import { Timestamp } from "@/components/shared/Timestamp";
import { useExecution, usePendingForExecution } from "@/hooks/use-execution";

export function ApprovalDetailPage() {
  const { execution_id } = useParams<{ execution_id: string }>();
  const { data: pending, isLoading, error } = usePendingForExecution(execution_id);
  // Once resolved, say who decided and when (1.89.0), not only "may have
  // already been resumed".
  const execution = useExecution(!pending && !isLoading ? execution_id : undefined);
  const resolution = execution.data?.resolution ?? null;

  if (!execution_id) {
    return null;
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="Approval"
        description={
          pending
            ? `Suspended at ${pending.node_id} · ${pending.reason}`
            : isLoading
              ? "Loading…"
              : resolution
                ? "This approval has been resolved."
                : "No pending interrupt for this execution."
        }
      >
        <Link
          to="/approvals"
          className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Approvals
        </Link>
      </PageHeader>

      {error && (
        <Card className="border-destructive/50">
          <CardContent className="pt-4 flex items-start gap-2 text-sm text-destructive">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{error instanceof Error ? error.message : String(error)}</span>
          </CardContent>
        </Card>
      )}

      {pending && (
        <>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Suspension</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <Field label="execution_id" value={pending.execution_id} mono />
                <Field label="chain_name" value={pending.chain_name} />
                <Field label="node_id" value={pending.node_id} mono />
                <Field label="reason" value={pending.reason} />
                <Field
                  label="agent_path"
                  value={pending.agent_path || "—"}
                  mono
                />
                <Field label="created_at" value={pending.created_at} mono />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-base">Frozen context</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="mb-2 text-xs text-muted-foreground">
                  This is the context dict passed to <code>interrupt()</code> at
                  suspend time. It is JSON-serialized into the database and
                  never recomputed — the human approves a specific snapshot.
                </p>
                <JsonViewer data={pending.context} className="max-h-72 p-3" />
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Decision</CardTitle>
            </CardHeader>
            <CardContent>
              <ApprovalActions executionId={pending.execution_id} />
            </CardContent>
          </Card>
        </>
      )}

      {!pending && !isLoading && !error && resolution && (
        <Card data-testid="approval-resolved">
          <CardContent className="space-y-2 py-6 text-center text-sm">
            <p>
              {resolution.approved === false
                ? "Rejected"
                : resolution.approved === true
                  ? "Approved"
                  : "Resolved"}
              {resolution.resolver ? (
                <>
                  {" "}by <span className="font-medium">{resolution.resolver}</span>
                </>
              ) : null}
              {resolution.at ? (
                <>
                  {" "}
                  <Timestamp iso={resolution.at} />
                </>
              ) : null}
              .
            </p>
            {resolution.reason && (
              <p className="text-xs text-muted-foreground">
                Reason: {resolution.reason}
              </p>
            )}
            <p className="text-xs">
              <Link
                to={`/executions/${execution_id}`}
                className="text-primary hover:underline"
              >
                View the execution&apos;s outcome
              </Link>
              {resolution.trace_id && (
                <>
                  {" · "}
                  <Link
                    to={`/traces/${resolution.trace_id}`}
                    className="text-primary hover:underline"
                  >
                    resumed run&apos;s trace
                  </Link>
                </>
              )}
            </p>
          </CardContent>
        </Card>
      )}

      {!pending && !isLoading && !error && !resolution && (
        <Card>
          <CardContent className="py-8 text-center text-sm text-muted-foreground">
            No pending interrupt for execution{" "}
            <code className="font-mono">{execution_id}</code>. It may have
            already been resumed.{" "}
            <Link
              to={`/executions/${execution_id}`}
              className="text-primary hover:underline"
            >
              View execution history
            </Link>
            .
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function Field({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="flex items-start gap-3">
      <span className="w-32 shrink-0 text-xs font-mono uppercase tracking-widest text-muted-foreground pt-0.5">
        {label}
      </span>
      <span className={mono ? "font-mono text-xs break-all" : "text-sm"}>
        {value}
      </span>
    </div>
  );
}
