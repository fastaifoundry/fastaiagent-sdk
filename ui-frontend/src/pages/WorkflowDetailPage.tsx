import { Link, useParams } from "react-router-dom";
import {
  ChevronLeft,
  GitBranch,
  Network,
  RefreshCw,
  UsersRound,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { StatCard } from "@/components/shared/StatCard";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { TracesTable } from "@/components/traces/TracesTable";
import { useTraces } from "@/hooks/use-traces";
import { useWorkflow } from "@/hooks/use-workflows";
import { formatCost, formatDurationMs, formatTimeAgo } from "@/lib/format";
import type { RunnerType } from "@/lib/types";

const ICON: Record<Exclude<RunnerType, "agent">, LucideIcon> = {
  chain: GitBranch,
  swarm: Network,
  supervisor: UsersRound,
};

export function WorkflowDetailPage() {
  const { runnerType, name } = useParams<{
    runnerType: string;
    name: string;
  }>();
  const workflow = useWorkflow(runnerType, name);
  const traces = useTraces({
    runner_type: (runnerType as RunnerType) ?? null,
    runner_name: name ?? null,
    page: 1,
    page_size: 50,
  });

  if (workflow.isLoading) return <TableSkeleton rows={4} />;
  if (workflow.error || !workflow.data) {
    return <EmptyState title="Workflow not found" icon={GitBranch} />;
  }
  const data = workflow.data;
  const Icon = ICON[data.runner_type] ?? GitBranch;

  return (
    <div className="space-y-5">
      <PageHeader
        title={data.workflow_name}
        description={`${data.runner_type}${data.node_count != null ? ` · ${data.node_count} nodes` : ""}`}
      >
        <Link to="/workflows">
          <Button variant="ghost" size="sm">
            <ChevronLeft className="mr-1.5 h-3.5 w-3.5" />
            Back
          </Button>
        </Link>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            workflow.refetch();
            traces.refetch();
          }}
          disabled={workflow.isFetching || traces.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${
              workflow.isFetching || traces.isFetching ? "animate-spin" : ""
            }`}
          />
          Refresh
        </Button>
      </PageHeader>

      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <Icon className="h-4 w-4" />
        <span>
          Runs of this {data.runner_type} produce one root span named{" "}
          <code>{data.runner_type}.{data.workflow_name}</code>.
        </span>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <StatCard label="Runs" value={String(data.run_count)} />
        <StatCard
          label="Success"
          value={`${Math.round(data.success_rate * 100)}%`}
        />
        <StatCard
          label="Avg latency"
          value={formatDurationMs(data.avg_latency_ms)}
        />
        <StatCard label="Avg cost" value={formatCost(data.avg_cost_usd)} />
      </div>

      <p className="text-xs text-muted-foreground">
        Last run {formatTimeAgo(data.last_run)} · {data.error_count} error
        {data.error_count === 1 ? "" : "s"} tracked.
      </p>

      {traces.isLoading ? (
        <TableSkeleton rows={6} />
      ) : (traces.data?.rows ?? []).length === 0 ? (
        <EmptyState
          title="No traces yet for this workflow"
          description="Run the workflow and refresh."
        />
      ) : (
        <TracesTable rows={traces.data?.rows ?? []} />
      )}
    </div>
  );
}
