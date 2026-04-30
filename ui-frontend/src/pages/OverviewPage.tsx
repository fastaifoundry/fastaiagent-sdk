import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { StatCard } from "@/components/shared/StatCard";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/shared/EmptyState";
import { api } from "@/lib/api";

interface OverviewPayload {
  traces_last_24h: number;
  failing_traces_last_24h: number;
  eval_runs_last_7d: number;
  avg_pass_rate_last_7d: number;
  pending_approvals_count: number;
  failed_executions_count: number;
  recent_traces: { trace_id: string; name: string; start_time: string; status: string }[];
  recent_eval_runs: {
    run_id: string;
    run_name: string;
    dataset_name: string;
    pass_rate: number;
    started_at: string;
  }[];
  prompt_changes_last_7d: { slug: string; version: string; created_at: string }[];
  agents_with_errors: { agent_name: string; error_count: number }[];
}

export function OverviewPage() {
  const { data, refetch, isFetching } = useQuery({
    queryKey: ["overview"],
    queryFn: () => api.get<OverviewPayload>("/overview"),
  });

  return (
    <div className="space-y-6">
      <PageHeader title="Home" description="What happened since you last looked.">
        <Button variant="outline" size="sm" onClick={() => refetch()} disabled={isFetching}>
          <RefreshCw className={`mr-1.5 h-3.5 w-3.5 ${isFetching ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </PageHeader>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <StatCard label="Traces (24h)" value={String(data?.traces_last_24h ?? "—")} />
        <StatCard
          label="Failing (24h)"
          value={String(data?.failing_traces_last_24h ?? "—")}
        />
        <StatCard label="Eval runs (7d)" value={String(data?.eval_runs_last_7d ?? "—")} />
        <StatCard
          label="Avg pass rate (7d)"
          value={
            data ? `${Math.round((data.avg_pass_rate_last_7d ?? 0) * 100)}%` : "—"
          }
        />
      </div>

      {/* v1.0 durability KPIs — paused workflows + failed/interrupted runs.
          Each card is wrapped in a Link so a click jumps straight to the
          relevant page. */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <Link to="/approvals" className="block transition-transform hover:scale-[1.01]">
          <StatCard
            label="Pending approvals"
            value={String(data?.pending_approvals_count ?? "—")}
          />
        </Link>
        <Link
          to="/approvals"
          className="block transition-transform hover:scale-[1.01]"
        >
          <StatCard
            label="Failed executions"
            value={String(data?.failed_executions_count ?? "—")}
          />
        </Link>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Recent traces</CardTitle>
          </CardHeader>
          <CardContent>
            {data?.recent_traces?.length ? (
              <ul className="space-y-2 text-sm">
                {data.recent_traces.map((t) => (
                  <li
                    key={t.trace_id}
                    className="flex items-center justify-between rounded-md px-2 py-1.5 hover:bg-muted/50"
                  >
                    <span className="truncate">{t.name || t.trace_id}</span>
                    <span className="ml-3 shrink-0 text-xs text-muted-foreground">
                      {t.status}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <EmptyState
                title="No traces yet"
                description="Run an agent — traces will appear here."
              />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Recent eval runs</CardTitle>
          </CardHeader>
          <CardContent>
            {data?.recent_eval_runs?.length ? (
              <ul className="space-y-2 text-sm">
                {data.recent_eval_runs.map((r) => (
                  <li
                    key={r.run_id}
                    className="flex items-center justify-between rounded-md px-2 py-1.5 hover:bg-muted/50"
                  >
                    <span className="truncate">{r.run_name || r.run_id}</span>
                    <span className="ml-3 shrink-0 text-xs text-muted-foreground">
                      {Math.round((r.pass_rate ?? 0) * 100)}%
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <EmptyState
                title="No eval runs yet"
                description="Call evaluate() — runs will appear here."
              />
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
