import { RefreshCw } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { PageHeader } from "@/components/layout/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
import { useOptimizeRuns } from "@/hooks/use-optimizes";
import { formatTimeAgo } from "@/lib/format";

const ALL_AGENTS = "__all__";

function fmtScore(n: number | null | undefined): string {
  return n != null ? n.toFixed(3) : "—";
}

export function OptimizeRunsPage() {
  const navigate = useNavigate();
  // Each optimize run targets a single agent; with several agents the list
  // interleaves their runs by recency. Filter server-side via ?agent= so the
  // list focuses on one agent. Options come from the unfiltered query (which
  // shares a cache entry with the display query while no filter is applied).
  const [agent, setAgent] = useState<string>(ALL_AGENTS);
  const isFiltered = agent !== ALL_AGENTS;
  const runs = useOptimizeRuns(isFiltered ? { agent } : {});
  const all = useOptimizeRuns({});
  const rows = runs.data?.rows ?? [];
  const agentOptions = useMemo(() => {
    const seen = new Set<string>();
    for (const r of all.data?.rows ?? []) {
      if (r.agent_name) seen.add(r.agent_name);
    }
    return Array.from(seen).sort();
  }, [all.data]);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Optimize runs"
        description={
          runs.data
            ? `${runs.data.total.toLocaleString()} run${runs.data.total === 1 ? "" : "s"}`
            : undefined
        }
      >
        {agentOptions.length > 1 && (
          <Select value={agent} onValueChange={setAgent}>
            <SelectTrigger className="w-[180px] text-xs">
              <SelectValue placeholder="All agents" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ALL_AGENTS}>All agents</SelectItem>
              {agentOptions.map((name) => (
                <SelectItem key={name} value={name}>
                  {name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            runs.refetch();
            all.refetch();
          }}
          disabled={runs.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${runs.isFetching ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </PageHeader>

      {runs.isLoading ? (
        <TableSkeleton rows={6} />
      ) : rows.length === 0 ? (
        <EmptyState
          title="No optimize runs yet"
          description="Call optimize(..., persist=True) — runs appear here automatically."
        />
      ) : (
        <div className="rounded-md border bg-card">
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead>Run</TableHead>
                <TableHead>Agent</TableHead>
                <TableHead>Levers</TableHead>
                <TableHead className="w-[100px]">Baseline</TableHead>
                <TableHead className="w-[100px]">Best</TableHead>
                <TableHead className="w-[120px]">Outcome</TableHead>
                <TableHead className="w-[120px]">Started</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => {
                const improved =
                  row.best_score != null &&
                  row.baseline_score != null &&
                  row.best_score > row.baseline_score &&
                  !row.reverted;
                return (
                  <TableRow
                    key={row.run_id}
                    className="cursor-pointer"
                    onClick={() => navigate(`/optimizes/${row.run_id}`)}
                  >
                    <TableCell className="font-medium">
                      {row.run_name || row.agent_name || row.run_id.slice(0, 8)}
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {row.agent_name ?? "—"}
                    </TableCell>
                    <TableCell className="text-sm">
                      {row.levers?.length ? row.levers.join(" → ") : "—"}
                    </TableCell>
                    <TableCell className="font-mono text-xs tabular-nums">
                      {fmtScore(row.baseline_score)}
                    </TableCell>
                    <TableCell className="font-mono text-xs tabular-nums">
                      {fmtScore(row.best_score)}
                    </TableCell>
                    <TableCell>
                      {row.reverted ? (
                        <Badge variant="destructive">reverted</Badge>
                      ) : improved ? (
                        <Badge variant="default">improved</Badge>
                      ) : (
                        <Badge variant="secondary">no change</Badge>
                      )}
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {formatTimeAgo(row.started_at)}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}
