import { Link } from "react-router-dom";
import { ExternalLink } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Timestamp } from "@/components/shared/Timestamp";
import type { ExecutionDetail, ExecutionRun } from "@/hooks/use-execution";
import { shortTraceId } from "@/lib/format";
import { cn } from "@/lib/utils";

const RUN_STATUS_TONE: Record<string, string> = {
  completed: "bg-fa-success/10 text-fa-success",
  paused: "bg-fa-warning/10 text-fa-warning",
  failed: "bg-destructive/10 text-destructive",
};

function asText(value: unknown): string {
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

/**
 * What happened to an execution (1.89.0, A1): the decision on its pause, who
 * made it and why, the answer, and every trace that ran it — the first run,
 * each resume, a fork. The checkpoint timeline only says what is stored.
 */
export function ExecutionOutcomeCard({ detail }: { detail: ExecutionDetail }) {
  const runs = detail.runs ?? [];
  const resolution = detail.resolution ?? null;
  const output = detail.final_output;
  if (!resolution && runs.length === 0 && output == null) return null;

  return (
    <Card data-testid="execution-outcome">
      <CardHeader>
        <CardTitle className="text-base">Outcome</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        {resolution ? (
          <div className="space-y-1" data-testid="execution-resolution">
            <p>
              <span
                className={cn(
                  "rounded px-2 py-0.5 text-xs font-medium",
                  resolution.approved === false
                    ? "bg-destructive/10 text-destructive"
                    : "bg-fa-success/10 text-fa-success"
                )}
              >
                {resolution.approved === false
                  ? "Rejected"
                  : resolution.approved === true
                    ? "Approved"
                    : "Resolved"}
              </span>
              {resolution.resolver && (
                <span className="ml-2">
                  by <span className="font-medium">{resolution.resolver}</span>
                </span>
              )}
              {resolution.at && (
                <span className="ml-2 text-xs text-muted-foreground">
                  <Timestamp iso={resolution.at} />
                </span>
              )}
            </p>
            {resolution.reason && (
              <p className="text-xs text-muted-foreground">
                Reason: <span className="text-foreground">{resolution.reason}</span>
              </p>
            )}
          </div>
        ) : detail.pending_interrupt ? (
          <p className="text-xs text-muted-foreground">
            Waiting for a decision at{" "}
            <span className="font-mono">{detail.pending_interrupt.node_id}</span>.
          </p>
        ) : null}

        {output != null && (
          <div className="space-y-1">
            <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
              Final output
            </div>
            <pre
              className="max-h-48 overflow-auto rounded-md border bg-muted/20 p-2 font-mono text-xs whitespace-pre-wrap"
              data-testid="execution-final-output"
            >
              {asText(output)}
            </pre>
          </div>
        )}

        {runs.length > 0 && (
          <div className="space-y-1">
            <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
              Runs
            </div>
            <ul className="space-y-1.5" data-testid="execution-runs">
              {runs.map((run) => (
                <RunRow key={run.trace_id} run={run} />
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function RunRow({ run }: { run: ExecutionRun }) {
  return (
    <li className="flex flex-wrap items-center gap-2 text-xs">
      <span className="w-14 font-mono uppercase text-muted-foreground">{run.role}</span>
      <span
        className={cn(
          "rounded px-1.5 py-0.5 text-[10px] font-medium",
          RUN_STATUS_TONE[run.status] ?? "bg-muted text-muted-foreground"
        )}
      >
        {run.status}
      </span>
      <Link
        to={`/traces/${run.trace_id}`}
        className="inline-flex items-center gap-1 font-mono text-primary hover:underline"
      >
        <ExternalLink className="h-3 w-3" />
        {shortTraceId(run.trace_id)}
      </Link>
      {run.started_at && (
        <span className="text-muted-foreground">
          <Timestamp iso={run.started_at} />
        </span>
      )}
    </li>
  );
}
