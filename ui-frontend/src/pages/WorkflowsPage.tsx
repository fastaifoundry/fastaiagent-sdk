import { useMemo, useState } from "react";
import {
  AlertTriangle,
  GitBranch,
  Network,
  RefreshCw,
  Search,
  UsersRound,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { DirectoryCard } from "@/components/shared/DirectoryCard";
import { useWorkflows } from "@/hooks/use-workflows";
import { formatCost, formatDurationMs, formatTimeAgo } from "@/lib/format";
import type { WorkflowSummary } from "@/lib/types";

type FilterType = "all" | "chain" | "swarm" | "supervisor";

const ICON: Record<Exclude<FilterType, "all">, LucideIcon> = {
  chain: GitBranch,
  swarm: Network,
  supervisor: UsersRound,
};

export function WorkflowsPage() {
  const [filter, setFilter] = useState<FilterType>("all");
  const [query, setQuery] = useState("");
  const workflows = useWorkflows(filter === "all" ? null : filter);
  const allRows = workflows.data?.workflows ?? [];
  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return allRows;
    return allRows.filter((w) =>
      w.workflow_name.toLowerCase().includes(needle)
    );
  }, [allRows, query]);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Workflows"
        description={
          workflows.data
            ? query
              ? `${rows.length} of ${allRows.length} workflows match "${query}"`
              : `${allRows.length} workflow${allRows.length === 1 ? "" : "s"} — chains, swarms, supervisors derived from trace roots`
            : undefined
        }
      >
        <div className="relative">
          <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            type="search"
            placeholder="Search workflows…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="h-8 w-56 pl-7 text-xs"
            aria-label="Search workflows"
          />
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => workflows.refetch()}
          disabled={workflows.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${workflows.isFetching ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </PageHeader>

      <Tabs value={filter} onValueChange={(v) => setFilter(v as FilterType)}>
        <TabsList>
          <TabsTrigger value="all">All</TabsTrigger>
          <TabsTrigger value="chain">
            <GitBranch className="mr-1.5 h-3.5 w-3.5" />
            Chains
          </TabsTrigger>
          <TabsTrigger value="swarm">
            <Network className="mr-1.5 h-3.5 w-3.5" />
            Swarms
          </TabsTrigger>
          <TabsTrigger value="supervisor">
            <UsersRound className="mr-1.5 h-3.5 w-3.5" />
            Supervisors
          </TabsTrigger>
        </TabsList>
      </Tabs>

      {workflows.isLoading ? (
        <TableSkeleton rows={4} />
      ) : allRows.length === 0 ? (
        <EmptyState
          title={
            filter === "all"
              ? "No workflows yet"
              : `No ${filter}s recorded`
          }
          icon={GitBranch}
          description="Run a Chain, Swarm, or Supervisor and a root span will land here."
        />
      ) : rows.length === 0 ? (
        <EmptyState
          title="No workflows match that search"
          icon={GitBranch}
          description={`Nothing matches "${query}". Clear the search to see all ${allRows.length}.`}
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {rows.map((wf) => (
            <WorkflowCard key={`${wf.runner_type}:${wf.workflow_name}`} wf={wf} />
          ))}
        </div>
      )}
    </div>
  );
}

function WorkflowCard({ wf }: { wf: WorkflowSummary }) {
  const Icon = ICON[wf.runner_type];
  const errorHeavy = wf.error_count > 0 && wf.success_rate < 0.9;
  return (
    <DirectoryCard
      to={`/workflows/${encodeURIComponent(wf.runner_type)}/${encodeURIComponent(wf.workflow_name)}`}
      icon={Icon}
      title={wf.workflow_name}
      badge={
        errorHeavy ? (
          <span title="Recent errors">
            <AlertTriangle className="h-3.5 w-3.5 text-destructive" />
          </span>
        ) : null
      }
      chip={
        <div className="inline-flex rounded bg-muted px-1.5 py-0.5 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
          {wf.runner_type}
          {wf.node_count != null && ` · ${wf.node_count} nodes`}
        </div>
      }
      stats={[
        { label: "Runs", value: String(wf.run_count) },
        {
          label: "Success",
          value: `${Math.round(wf.success_rate * 100)}%`,
          accent:
            wf.success_rate >= 0.9
              ? "text-fa-success"
              : wf.success_rate >= 0.7
              ? "text-fa-warning"
              : "text-destructive",
        },
        { label: "Avg latency", value: formatDurationMs(wf.avg_latency_ms) },
        { label: "Avg cost", value: formatCost(wf.avg_cost_usd) },
      ]}
      footer={
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>Last run</span>
          <span className="font-mono">{formatTimeAgo(wf.last_run)}</span>
        </div>
      }
    />
  );
}
