import { useState } from "react";
import { Link } from "react-router-dom";
import {
  AlertTriangle,
  GitBranch,
  Network,
  RefreshCw,
  UsersRound,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { useWorkflows } from "@/hooks/use-workflows";
import { formatCost, formatDurationMs, formatTimeAgo } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { WorkflowSummary } from "@/lib/types";

type FilterType = "all" | "chain" | "swarm" | "supervisor";

const ICON: Record<Exclude<FilterType, "all">, LucideIcon> = {
  chain: GitBranch,
  swarm: Network,
  supervisor: UsersRound,
};

export function WorkflowsPage() {
  const [filter, setFilter] = useState<FilterType>("all");
  const workflows = useWorkflows(filter === "all" ? null : filter);
  const rows = workflows.data?.workflows ?? [];

  return (
    <div className="space-y-5">
      <PageHeader
        title="Workflows"
        description={
          workflows.data
            ? `${rows.length} workflow${rows.length === 1 ? "" : "s"} — chains, swarms, supervisors derived from trace roots`
            : undefined
        }
      >
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
      ) : rows.length === 0 ? (
        <EmptyState
          title={
            filter === "all"
              ? "No workflows yet"
              : `No ${filter}s recorded`
          }
          icon={GitBranch}
          description="Run a Chain, Swarm, or Supervisor and a root span will land here."
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
    <Link
      to={`/workflows/${encodeURIComponent(wf.runner_type)}/${encodeURIComponent(wf.workflow_name)}`}
    >
      <Card className="h-full transition-colors hover:border-primary">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center justify-between text-sm">
            <span className="inline-flex items-center gap-2 truncate">
              <Icon className="h-4 w-4 shrink-0 text-muted-foreground" />
              {wf.workflow_name}
            </span>
            {errorHeavy && (
              <span title="Recent errors">
                <AlertTriangle className="h-3.5 w-3.5 text-destructive" />
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 pt-0">
          <div className="inline-flex rounded bg-muted px-1.5 py-0.5 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
            {wf.runner_type}
            {wf.node_count != null && ` · ${wf.node_count} nodes`}
          </div>
          <dl className="grid grid-cols-2 gap-2 text-xs">
            <Stat label="Runs" value={String(wf.run_count)} />
            <Stat
              label="Success"
              value={`${Math.round(wf.success_rate * 100)}%`}
              accent={
                wf.success_rate >= 0.9
                  ? "text-fa-success"
                  : wf.success_rate >= 0.7
                  ? "text-fa-warning"
                  : "text-destructive"
              }
            />
            <Stat label="Avg latency" value={formatDurationMs(wf.avg_latency_ms)} />
            <Stat label="Avg cost" value={formatCost(wf.avg_cost_usd)} />
          </dl>
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Last run</span>
            <span className="font-mono">{formatTimeAgo(wf.last_run)}</span>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}

function Stat({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div>
      <dt className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        {label}
      </dt>
      <dd className={cn("font-mono text-sm tabular-nums", accent)}>{value}</dd>
    </div>
  );
}
