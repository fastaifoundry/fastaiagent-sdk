import { lazy, Suspense } from "react";
import { Link, useParams } from "react-router-dom";
import { Bot, ChevronLeft, GitBranch, RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { StatCard } from "@/components/shared/StatCard";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { TracesTable } from "@/components/traces/TracesTable";
import { AgentToolsSection } from "@/components/agents/AgentToolsSection";
import { useAgent, useAgentTools } from "@/hooks/use-agents";
import { useTraces } from "@/hooks/use-traces";
import { formatCost, formatDurationMs, formatTimeAgo } from "@/lib/format";

const WorkflowTopologyView = lazy(() =>
  import("@/components/workflows/WorkflowTopologyView").then((m) => ({
    default: m.WorkflowTopologyView,
  }))
);

export function AgentDetailPage() {
  const { name } = useParams<{ name: string }>();
  const agent = useAgent(name);
  const traces = useTraces({ agent: name ?? null, page: 1, page_size: 50 });
  const tools = useAgentTools(name);

  if (agent.isLoading) return <TableSkeleton rows={4} />;
  if (agent.error || !agent.data) {
    return <EmptyState title="Agent not found" icon={Bot} />;
  }

  const rows = traces.data?.rows ?? [];

  return (
    <div className="space-y-5">
      <PageHeader title={agent.data.agent_name} description="Agent detail">
        <Link to="/agents">
          <Button variant="ghost" size="sm">
            <ChevronLeft className="mr-1.5 h-3.5 w-3.5" />
            Back
          </Button>
        </Link>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            agent.refetch();
            traces.refetch();
            tools.refetch();
          }}
          disabled={agent.isFetching || traces.isFetching || tools.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${
              agent.isFetching || traces.isFetching || tools.isFetching
                ? "animate-spin"
                : ""
            }`}
          />
          Refresh
        </Button>
      </PageHeader>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <StatCard label="Runs" value={String(agent.data.run_count)} />
        <StatCard
          label="Success"
          value={`${Math.round(agent.data.success_rate * 100)}%`}
        />
        <StatCard
          label="Avg latency"
          value={formatDurationMs(agent.data.avg_latency_ms)}
        />
        <StatCard
          label="Avg cost"
          value={formatCost(agent.data.avg_cost_usd)}
        />
      </div>

      <p className="text-xs text-muted-foreground">
        Last run {formatTimeAgo(agent.data.last_run)} · {agent.data.error_count}{" "}
        error{agent.data.error_count === 1 ? "" : "s"} tracked.
      </p>

      {(agent.data.workflows ?? []).length > 0 ? (
        <div>
          <h2 className="mb-2 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            // PARTICIPATES IN
          </h2>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            {(agent.data.workflows ?? []).map((wf) => (
              <Link
                key={`${wf.runner_type}:${wf.name}`}
                to={`/workflows/${wf.runner_type}/${encodeURIComponent(wf.name)}`}
                className="block rounded-md border bg-card hover:border-primary/40 transition-colors"
              >
                <div className="flex items-center justify-between px-3 py-2 border-b border-border">
                  <div className="flex items-center gap-1.5">
                    <GitBranch className="h-3.5 w-3.5 text-primary" />
                    <span className="font-mono text-xs">{wf.name}</span>
                  </div>
                  <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-widest">
                    {wf.runner_type}
                  </span>
                </div>
                <Suspense
                  fallback={
                    <div className="px-3 py-6 text-xs text-muted-foreground">
                      Loading…
                    </div>
                  }
                >
                  <WorkflowTopologyView
                    runnerType={wf.runner_type}
                    name={wf.name}
                    compact
                    height={180}
                  />
                </Suspense>
              </Link>
            ))}
          </div>
        </div>
      ) : null}

      <AgentToolsSection data={tools.data} />

      {traces.isLoading ? (
        <TableSkeleton rows={6} />
      ) : rows.length === 0 ? (
        <EmptyState
          title="No traces yet for this agent"
          description="Run the agent and refresh."
        />
      ) : (
        <TracesTable rows={rows} />
      )}
    </div>
  );
}
