import { Link, useParams } from "react-router-dom";
import { ChevronLeft, ExternalLink, Play, RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { StatCard } from "@/components/shared/StatCard";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { PassRateBar } from "@/components/evals/PassRateBar";
import { useEvalRun } from "@/hooks/use-evals";
import { formatTimeAgo } from "@/lib/format";
import { cn } from "@/lib/utils";

function stringifyShort(value: unknown, max = 120): string {
  if (value == null) return "—";
  if (typeof value === "string") {
    return value.length > max ? `${value.slice(0, max)}…` : value;
  }
  try {
    const s = JSON.stringify(value);
    return s.length > max ? `${s.slice(0, max)}…` : s;
  } catch {
    return String(value);
  }
}

export function EvalRunDetailPage() {
  const { runId } = useParams<{ runId: string }>();
  const query = useEvalRun(runId);

  if (query.isLoading) return <TableSkeleton rows={8} />;
  if (query.error || !query.data) {
    return (
      <EmptyState
        title="Eval run not found"
        description="This run id doesn't exist in the local DB."
      />
    );
  }

  const { run, cases } = query.data;
  const scorerNames = Array.from(
    new Set(cases.flatMap((c) => Object.keys(c.per_scorer ?? {})))
  );

  return (
    <div className="space-y-5">
      <PageHeader
        title={run.run_name || `Run ${run.run_id.slice(0, 8)}`}
        description={`Dataset ${run.dataset_name ?? "—"} · Agent ${run.agent_name ?? "—"}`}
      >
        <Link to="/evals">
          <Button variant="ghost" size="sm">
            <ChevronLeft className="mr-1.5 h-3.5 w-3.5" />
            Back
          </Button>
        </Link>
        <Button
          variant="outline"
          size="sm"
          onClick={() => query.refetch()}
          disabled={query.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${query.isFetching ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </PageHeader>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <StatCard label="Pass rate" value={`${Math.round((run.pass_rate ?? 0) * 100)}%`} />
        <StatCard label="Passed" value={String(run.pass_count ?? 0)} />
        <StatCard label="Failed" value={String(run.fail_count ?? 0)} />
        <StatCard
          label="Started"
          value={formatTimeAgo(run.started_at ?? "")}
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Cases</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="w-[50px]">#</TableHead>
                <TableHead>Input</TableHead>
                <TableHead>Expected</TableHead>
                <TableHead>Actual</TableHead>
                {scorerNames.map((name) => (
                  <TableHead key={name} className="w-[120px]">
                    {name}
                  </TableHead>
                ))}
                <TableHead className="w-[48px]" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {cases.map((c) => (
                <TableRow key={c.case_id}>
                  <TableCell className="font-mono text-xs text-muted-foreground">
                    {c.ordinal}
                  </TableCell>
                  <TableCell
                    className="max-w-[200px] truncate font-mono text-xs"
                    title={stringifyShort(c.input, 2000)}
                  >
                    {stringifyShort(c.input)}
                  </TableCell>
                  <TableCell
                    className="max-w-[200px] truncate font-mono text-xs"
                    title={stringifyShort(c.expected_output, 2000)}
                  >
                    {stringifyShort(c.expected_output)}
                  </TableCell>
                  <TableCell
                    className="max-w-[200px] truncate font-mono text-xs"
                    title={stringifyShort(c.actual_output, 2000)}
                  >
                    {stringifyShort(c.actual_output)}
                  </TableCell>
                  {scorerNames.map((name) => {
                    const score = c.per_scorer?.[name];
                    if (!score)
                      return (
                        <TableCell key={name} className="text-xs text-muted-foreground">
                          —
                        </TableCell>
                      );
                    return (
                      <TableCell key={name}>
                        <span
                          className={cn(
                            "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs font-mono",
                            score.passed
                              ? "bg-fa-success/10 text-fa-success"
                              : "bg-destructive/10 text-destructive"
                          )}
                        >
                          <span className="h-1.5 w-1.5 rounded-full bg-current" />
                          {score.passed ? "pass" : "fail"}
                          <span className="tabular-nums opacity-70">
                            {(score.score ?? 0).toFixed(2)}
                          </span>
                        </span>
                      </TableCell>
                    );
                  })}
                  <TableCell>
                    {c.trace_id && (
                      <Link
                        to={`/traces/${c.trace_id}/replay`}
                        title="Open in Replay"
                        className="inline-flex items-center text-muted-foreground hover:text-primary"
                      >
                        <Play className="h-3.5 w-3.5" />
                      </Link>
                    )}
                    {c.trace_id && (
                      <Link
                        to={`/traces/${c.trace_id}`}
                        title="Open trace"
                        className="ml-1 inline-flex items-center text-muted-foreground hover:text-primary"
                      >
                        <ExternalLink className="h-3.5 w-3.5" />
                      </Link>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          {cases.length > 0 && (
            <div className="border-t p-3">
              <PassRateBar
                passRate={run.pass_rate}
                passCount={run.pass_count}
                failCount={run.fail_count}
              />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
