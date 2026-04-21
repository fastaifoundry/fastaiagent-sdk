import { AlertTriangle, Bot, RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { DirectoryCard } from "@/components/shared/DirectoryCard";
import { useAgents } from "@/hooks/use-agents";
import { formatCost, formatDurationMs, formatTimeAgo } from "@/lib/format";

export function AgentsPage() {
  const agents = useAgents();
  const rows = agents.data?.agents ?? [];

  return (
    <div className="space-y-5">
      <PageHeader
        title="Agents"
        description={
          agents.data
            ? `${rows.length} agent${rows.length === 1 ? "" : "s"} seen`
            : undefined
        }
      >
        <Button
          variant="outline"
          size="sm"
          onClick={() => agents.refetch()}
          disabled={agents.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${agents.isFetching ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </PageHeader>

      {agents.isLoading ? (
        <TableSkeleton rows={4} />
      ) : rows.length === 0 ? (
        <EmptyState
          title="No agents yet"
          icon={Bot}
          description="Agent definitions live in code. Once an agent runs and a trace lands, it'll show up here."
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {rows.map((agent) => {
            const errorHeavy = agent.error_count > 0 && agent.success_rate < 0.9;
            return (
              <DirectoryCard
                key={agent.agent_name}
                to={`/agents/${encodeURIComponent(agent.agent_name)}`}
                icon={Bot}
                title={agent.agent_name}
                badge={
                  errorHeavy ? (
                    <span title="Recent errors">
                      <AlertTriangle className="h-3.5 w-3.5 text-destructive" />
                    </span>
                  ) : null
                }
                stats={[
                  { label: "Runs", value: agent.run_count.toString() },
                  {
                    label: "Success",
                    value: `${Math.round(agent.success_rate * 100)}%`,
                    accent:
                      agent.success_rate >= 0.9
                        ? "text-fa-success"
                        : agent.success_rate >= 0.7
                        ? "text-fa-warning"
                        : "text-destructive",
                  },
                  {
                    label: "Avg latency",
                    value: formatDurationMs(agent.avg_latency_ms),
                  },
                  { label: "Avg cost", value: formatCost(agent.avg_cost_usd) },
                ]}
                footer={
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>Last run</span>
                    <span className="font-mono">
                      {formatTimeAgo(agent.last_run)}
                    </span>
                  </div>
                }
              />
            );
          })}
        </div>
      )}
    </div>
  );
}
