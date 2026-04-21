import { Link, useParams } from "react-router-dom";
import { Bot, ChevronLeft, RefreshCw } from "lucide-react";
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
