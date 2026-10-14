import { RefreshCw } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
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
import { useSimulationRuns } from "@/hooks/use-simulations";
import { Timestamp } from "@/components/shared/Timestamp";

export function SimulationsPage() {
  const navigate = useNavigate();
  const runs = useSimulationRuns();
  const rows = runs.data?.rows ?? [];

  return (
    <div className="space-y-5">
      <PageHeader
        title="Simulations"
        description={
          runs.data
            ? `${runs.data.total.toLocaleString()} run${
                runs.data.total === 1 ? "" : "s"
              }`
            : undefined
        }
      >
        <Button
          variant="outline"
          size="sm"
          onClick={() => runs.refetch()}
          disabled={runs.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${
              runs.isFetching ? "animate-spin" : ""
            }`}
          />
          Refresh
        </Button>
      </PageHeader>

      {runs.isLoading ? (
        <TableSkeleton rows={6} />
      ) : rows.length === 0 ? (
        <EmptyState
          title="No simulations yet"
          description="Call simulate() — scenario runs appear here automatically."
        />
      ) : (
        <div className="rounded-md border bg-card">
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead>Run</TableHead>
                <TableHead>Agent</TableHead>
                <TableHead className="w-[110px]">Scenarios</TableHead>
                <TableHead className="w-[200px]">Pass rate</TableHead>
                <TableHead
                  className="w-[90px]"
                  title="Scenarios that did not finish, or that the judge could not run on"
                >
                  Errored
                </TableHead>
                <TableHead className="w-[120px]">Started</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => (
                <TableRow
                  key={row.run_id}
                  className="cursor-pointer"
                  onClick={() => navigate(`/simulations/${row.run_id}`)}
                >
                  <TableCell className="font-medium">
                    {row.run_name || row.run_id}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {row.agent_name ?? "—"}
                  </TableCell>
                  <TableCell className="font-mono text-xs tabular-nums">
                    {row.scenario_count ?? 0}
                  </TableCell>
                  <TableCell>
                    <PassRateBar
                      passRate={row.pass_rate}
                      passCount={row.pass_count}
                      // The rate is over all scenarios, so the fraction is too. Not
                      // fail_count: it excludes errored scenarios since 1.89.0.
                      failCount={
                        row.scenario_count != null
                          ? Math.max(0, row.scenario_count - (row.pass_count ?? 0))
                          : row.fail_count
                      }
                      title="Passed / all scenarios. An errored scenario counts as not passed."
                    />
                  </TableCell>
                  <TableCell
                    className="font-mono text-xs tabular-nums"
                    data-testid={`sim-errored-${row.run_id}`}
                  >
                    {row.errored_count ? (
                      row.errored_count
                    ) : (
                      <span className="text-muted-foreground">{row.errored_count ?? "—"}</span>
                    )}
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    <Timestamp iso={row.started_at} />
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
