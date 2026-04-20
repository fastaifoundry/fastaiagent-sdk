import { Link } from "react-router-dom";
import { AlertTriangle, Bot, RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { useAgents } from "@/hooks/use-agents";
import { formatCost, formatDurationMs, formatTimeAgo } from "@/lib/format";
import { cn } from "@/lib/utils";

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
              <Link key={agent.agent_name} to={`/agents/${encodeURIComponent(agent.agent_name)}`}>
                <Card className="h-full transition-colors hover:border-primary">
                  <CardHeader className="pb-3">
                    <CardTitle className="flex items-center justify-between text-sm">
                      <span className="inline-flex items-center gap-2 truncate">
                        <Bot className="h-4 w-4 shrink-0 text-muted-foreground" />
                        {agent.agent_name}
                      </span>
                      {errorHeavy && (
                        <span title="Recent errors">
                          <AlertTriangle className="h-3.5 w-3.5 text-destructive" />
                        </span>
                      )}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3 pt-0">
                    <dl className="grid grid-cols-2 gap-2 text-xs">
                      <Stat label="Runs" value={agent.run_count.toString()} />
                      <Stat
                        label="Success"
                        value={`${Math.round(agent.success_rate * 100)}%`}
                        accent={
                          agent.success_rate >= 0.9
                            ? "text-fa-success"
                            : agent.success_rate >= 0.7
                            ? "text-fa-warning"
                            : "text-destructive"
                        }
                      />
                      <Stat label="Avg latency" value={formatDurationMs(agent.avg_latency_ms)} />
                      <Stat label="Avg cost" value={formatCost(agent.avg_cost_usd)} />
                    </dl>
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>Last run</span>
                      <span className="font-mono">{formatTimeAgo(agent.last_run)}</span>
                    </div>
                  </CardContent>
                </Card>
              </Link>
            );
          })}
        </div>
      )}
    </div>
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
