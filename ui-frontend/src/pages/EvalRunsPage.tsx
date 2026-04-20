import { RefreshCw } from "lucide-react";
import { useNavigate } from "react-router-dom";
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
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { PassRateBar } from "@/components/evals/PassRateBar";
import { QualityTrendChart } from "@/components/evals/QualityTrendChart";
import { useEvalRuns, useEvalTrend } from "@/hooks/use-evals";
import { formatTimeAgo } from "@/lib/format";

export function EvalRunsPage() {
  const navigate = useNavigate();
  const runs = useEvalRuns();
  const trend = useEvalTrend();

  const rows = runs.data?.rows ?? [];

  return (
    <div className="space-y-5">
      <PageHeader
        title="Eval runs"
        description={
          runs.data
            ? `${runs.data.total.toLocaleString()} run${runs.data.total === 1 ? "" : "s"}`
            : undefined
        }
      >
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            runs.refetch();
            trend.refetch();
          }}
          disabled={runs.isFetching || trend.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${
              runs.isFetching || trend.isFetching ? "animate-spin" : ""
            }`}
          />
          Refresh
        </Button>
      </PageHeader>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Quality trend</CardTitle>
        </CardHeader>
        <CardContent>
          {trend.isLoading ? (
            <TableSkeleton rows={3} />
          ) : (
            <QualityTrendChart points={trend.data?.points ?? []} />
          )}
        </CardContent>
      </Card>

      {runs.isLoading ? (
        <TableSkeleton rows={6} />
      ) : rows.length === 0 ? (
        <EmptyState
          title="No eval runs yet"
          description="Call evaluate() — results appear here automatically."
        />
      ) : (
        <div className="rounded-md border bg-card">
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead>Run</TableHead>
                <TableHead>Dataset</TableHead>
                <TableHead>Agent</TableHead>
                <TableHead>Scorers</TableHead>
                <TableHead className="w-[200px]">Pass rate</TableHead>
                <TableHead className="w-[120px]">Started</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => (
                <TableRow
                  key={row.run_id}
                  className="cursor-pointer"
                  onClick={() => navigate(`/evals/${row.run_id}`)}
                >
                  <TableCell className="font-medium">
                    {row.run_name || row.run_id}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {row.dataset_name ?? "—"}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {row.agent_name ?? "—"}
                  </TableCell>
                  <TableCell className="text-sm">
                    {row.scorers?.length
                      ? row.scorers.join(", ")
                      : "—"}
                  </TableCell>
                  <TableCell>
                    <PassRateBar
                      passRate={row.pass_rate}
                      passCount={row.pass_count}
                      failCount={row.fail_count}
                    />
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {formatTimeAgo(row.started_at)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}
