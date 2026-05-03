import { lazy, Suspense } from "react";
import { Link, useParams } from "react-router-dom";
import { Bot, ChevronLeft, GitBranch, RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { StatCard } from "@/components/shared/StatCard";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
const AgentDependencyGraph = lazy(() =>
  import("@/components/agents/AgentDependencyGraph").then((m) => ({
    default: m.AgentDependencyGraph,
  }))
);

export function AgentDetailPage() {
  const { name } = useParams<{ name: string }>();
  const agent = useAgent(name);
  const traces = useTraces({ agent: name ?? null, page: 1, page_size: 50 });
  const tools = useAgentTools(name);

  if (agent.isLoading) return <TableSkeleton rows={4} />;
  // 404 from /api/agents/<name> just means no runtime spans yet — for
  // registered-but-not-run agents the Dependencies tab still has data.
  // Synthesize an empty summary so the page renders.
  const summary = agent.data ?? {
    agent_name: name ?? "",
    run_count: 0,
    success_rate: 0,
    error_count: 0,
    avg_latency_ms: 0,
    avg_cost_usd: 0,
    last_run: "",
    workflows: [],
  };
  if (!name) {
    return <EmptyState title="Agent not found" icon={Bot} />;
  }

  const rows = traces.data?.rows ?? [];
  const noRunsYet = !agent.data;

  return (
    <div className="space-y-5">
      <PageHeader
        title={summary.agent_name}
        description={noRunsYet ? "Registered runner — no runs yet" : "Agent detail"}
      >
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
        <StatCard label="Runs" value={String(summary.run_count)} />
        <StatCard
          label="Success"
          value={`${Math.round(summary.success_rate * 100)}%`}
        />
        <StatCard
          label="Avg latency"
          value={formatDurationMs(summary.avg_latency_ms)}
        />
        <StatCard
          label="Avg cost"
          value={formatCost(summary.avg_cost_usd)}
        />
      </div>

      <p className="text-xs text-muted-foreground">
        Last run {formatTimeAgo(summary.last_run)} · {summary.error_count}{" "}
        error{summary.error_count === 1 ? "" : "s"} tracked.
      </p>

      {(summary.workflows ?? []).length > 0 ? (
        <div>
          <h2 className="mb-2 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            // PARTICIPATES IN
          </h2>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            {(summary.workflows ?? []).map((wf) => (
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

      <Tabs defaultValue="tools" className="w-full">
        <TabsList>
          <TabsTrigger value="tools">Tools</TabsTrigger>
          <TabsTrigger value="dependencies">Dependencies</TabsTrigger>
          <TabsTrigger value="runs">Run history</TabsTrigger>
        </TabsList>
        <TabsContent value="tools">
          <AgentToolsSection data={tools.data} />
        </TabsContent>
        <TabsContent value="dependencies">
          <Suspense
            fallback={
              <div className="px-3 py-6 text-xs text-muted-foreground">
                Loading dependency graph…
              </div>
            }
          >
            {name && <AgentDependencyGraph agentName={name} />}
          </Suspense>
        </TabsContent>
        <TabsContent value="runs">
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
        </TabsContent>
      </Tabs>
    </div>
  );
}
