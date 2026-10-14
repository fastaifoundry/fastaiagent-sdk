import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { FileText, RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { StatCard } from "@/components/shared/StatCard";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/shared/EmptyState";
import { TraceStatusBadge } from "@/components/traces/TraceStatusBadge";
import { Panel } from "@/components/charts/chart-kit";
import { AttentionStrip } from "@/components/overview/AttentionStrip";
import { buildAttention } from "@/components/overview/attention";
import {
  ExperimentsNote,
  IncludeExperimentsToggle,
} from "@/components/shared/ExperimentsNote";
import { GateBadge } from "@/components/evals/AgentCiPanel";
import { api } from "@/lib/api";
import { formatRate, KIND_LABEL } from "@/lib/evalDisplay";
import { useIncludeExperiments } from "@/stores/preferences-store";

interface OverviewPayload {
  traces_last_24h: number;
  // 1.89.0: experiment traces (replay, Playground, eval…) in the last 24h,
  // and whether the other counts include them.
  experiments_last_24h?: number;
  experiments_included?: boolean;
  failing_traces_last_24h: number;
  eval_runs_last_7d: number;
  /** Legacy: checks rate averaged over every run kind. Not shown. */
  avg_pass_rate_last_7d: number;
  // 1.89.0: cases rate over the latest run per dataset, quality runs only.
  eval_case_pass_rate_7d?: number | null;
  eval_datasets_7d?: number;
  eval_runs_counted_7d?: number;
  eval_runs_excluded_7d?: number;
  eval_gate_failures_7d?: number;
  pending_approvals_count: number;
  failed_executions_count: number;
  recent_traces: { trace_id: string; name: string; start_time: string; status: string }[];
  recent_eval_runs: {
    run_id: string;
    run_name: string;
    dataset_name: string;
    pass_rate: number | null;
    started_at: string;
    kind?: string | null;
    case_pass_rate?: number | null;
    gate_outcome?: string | null;
  }[];
  prompt_changes_last_7d: { slug: string; version: string; created_at: string }[];
  agents_with_errors: { agent_name: string; error_count: number }[];
}

/**
 * Pass-rate chip. Thresholds mirror the Enterprise console's score chips so a
 * given rate reads the same in both products: ≥85% good, ≥70% watch, below
 * that a failure. Always carries the number, never colour alone.
 */
function ScoreChip({ value, title }: { value: number | null | undefined; title?: string }) {
  if (typeof value !== "number") {
    return (
      <span
        className="shrink-0 rounded-full bg-muted px-2 py-0.5 font-mono text-[10.5px] font-medium text-muted-foreground"
        title={title ?? "No case reached a verdict"}
      >
        —
      </span>
    );
  }
  const pct = Math.round(value * 100);
  const tone =
    value >= 0.85
      ? "bg-fa-success/15 text-fa-success"
      : value >= 0.7
      ? "bg-fa-warning/15 text-fa-warning"
      : "bg-destructive/15 text-destructive";
  return (
    <span
      className={`shrink-0 rounded-full px-2 py-0.5 font-mono text-[10.5px] font-medium ${tone}`}
      title={title}
    >
      {pct}%
    </span>
  );
}

export function OverviewPage() {
  const includeExperiments = useIncludeExperiments();
  const { data, refetch, isFetching, isLoading } = useQuery({
    queryKey: ["overview", includeExperiments],
    queryFn: () =>
      api.get<OverviewPayload>(`/overview?include_experiments=${includeExperiments}`),
  });

  const attention = useMemo(() => buildAttention(data), [data]);

  return (
    <div className="space-y-6">
      <PageHeader title="Home" description="What happened since you last looked.">
        <IncludeExperimentsToggle />
        <Button variant="outline" size="sm" onClick={() => refetch()} disabled={isFetching}>
          <RefreshCw className={`mr-1.5 h-3.5 w-3.5 ${isFetching ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </PageHeader>

      {/* What's wrong, before what exists. */}
      <AttentionStrip items={attention} isLoading={isLoading} />

      {/* The counts below leave experiments out unless asked; say how many. */}
      <ExperimentsNote
        count={data?.experiments_last_24h}
        included={data?.experiments_included ?? false}
        suffix="from the last 24h"
        className="-mt-3"
      />
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <StatCard label="Traces (24h)" value={String(data?.traces_last_24h ?? "—")} />
        <StatCard
          label="Failing (24h)"
          value={String(data?.failing_traces_last_24h ?? "—")}
        />
        <StatCard label="Eval runs (7d)" value={String(data?.eval_runs_last_7d ?? "—")} />
        <StatCard
          label="Eval cases passed (7d)"
          value={formatRate(data?.eval_case_pass_rate_7d)}
          title="Cases pass rate, averaged over the latest valid run of each dataset. Dataset checks, AutoLLM candidate runs and invalid (outage) runs are left out."
          hint={
            data
              ? `${data.eval_datasets_7d ?? 0} dataset${data.eval_datasets_7d === 1 ? "" : "s"} · ${
                  data.eval_runs_counted_7d ?? 0
                } counted${
                  data.eval_runs_excluded_7d ? `, ${data.eval_runs_excluded_7d} left out` : ""
                }`
              : undefined
          }
        />
      </div>

      {/* v1.0 durability KPIs — paused workflows + failed/interrupted runs.
          Each card is wrapped in a Link so a click jumps straight to the
          relevant page. */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <Link
          to="/approvals"
          className="block rounded-xl transition-colors hover:border-primary/50 [&>div]:hover:border-primary/50"
        >
          <StatCard
            label="Pending approvals"
            value={String(data?.pending_approvals_count ?? "—")}
          />
        </Link>
        <Link
          to="/approvals"
          className="block rounded-xl transition-colors [&>div]:hover:border-primary/50"
        >
          <StatCard
            label="Failed executions"
            value={String(data?.failed_executions_count ?? "—")}
          />
        </Link>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Panel title="Recent traces" subtitle="Latest agent runs recorded locally">
          {data?.recent_traces?.length ? (
            <ul className="fa-body divide-y divide-border/60">
              {data.recent_traces.map((t) => (
                <li
                  key={t.trace_id}
                  className="flex items-center justify-between gap-3 py-2"
                >
                  <Link
                    to={`/traces/${t.trace_id}`}
                    className="truncate hover:text-primary hover:underline"
                  >
                    {t.name || t.trace_id}
                  </Link>
                  <TraceStatusBadge status={t.status} className="shrink-0" />
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState
              title="No traces yet"
              description="Run an agent — traces will appear here."
            />
          )}
        </Panel>

        <Panel title="Recent eval runs" subtitle="Most recent scored runs">
          {data?.recent_eval_runs?.length ? (
            <ul className="fa-body divide-y divide-border/60">
              {data.recent_eval_runs.map((r) => (
                <li
                  key={r.run_id}
                  className="flex items-center justify-between gap-3 py-2"
                >
                  <Link
                    to={`/evals/${r.run_id}`}
                    className="truncate hover:text-primary hover:underline"
                  >
                    {r.run_name || r.run_id}
                  </Link>
                  <span className="flex shrink-0 items-center gap-1.5">
                    {r.kind && KIND_LABEL[r.kind] && r.kind !== "evaluate" && (
                      <span className="font-mono text-[10.5px] text-muted-foreground">
                        {KIND_LABEL[r.kind]}
                      </span>
                    )}
                    {r.gate_outcome && <GateBadge outcome={r.gate_outcome} />}
                    <ScoreChip
                      value={r.case_pass_rate}
                      title={`Cases ${formatRate(r.case_pass_rate)} · Checks ${formatRate(r.pass_rate)}`}
                    />
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
        </Panel>
      </div>

      {/* Both of these were already in the /overview payload and rendered
          nowhere. */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Panel
          title="Agents in recent failures"
          subtitle="Grouped from the 10 most recent errors — a sample, not a total"
        >
          {data?.agents_with_errors?.length ? (
            <ul className="fa-body divide-y divide-border/60">
              {data.agents_with_errors.map((a) => (
                <li
                  key={a.agent_name}
                  className="flex items-center justify-between gap-3 py-2"
                >
                  <Link
                    to={`/agents/${encodeURIComponent(a.agent_name)}`}
                    className="truncate font-mono hover:text-primary hover:underline"
                  >
                    {a.agent_name}
                  </Link>
                  <span className="shrink-0 rounded-full bg-destructive/15 px-2 py-0.5 font-mono text-[10.5px] font-medium text-destructive">
                    {a.error_count} {a.error_count === 1 ? "error" : "errors"}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="py-6 text-center text-[12.5px] text-muted-foreground">
              No agent errors in the last 24 hours.
            </p>
          )}
        </Panel>

        <Panel
          title="Prompt changes"
          subtitle="New prompt versions committed in the last 7 days"
        >
          {data?.prompt_changes_last_7d?.length ? (
            <ul className="fa-body divide-y divide-border/60">
              {data.prompt_changes_last_7d.map((p) => (
                <li
                  key={`${p.slug}@${p.version}`}
                  className="flex items-center justify-between gap-3 py-2"
                >
                  <Link
                    to={`/prompts/${encodeURIComponent(p.slug)}`}
                    className="flex min-w-0 items-center gap-2 hover:text-primary"
                  >
                    <FileText className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                    <span className="truncate font-mono hover:underline">{p.slug}</span>
                  </Link>
                  <span className="shrink-0 font-mono text-[11px] text-muted-foreground">
                    v{p.version}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="py-6 text-center text-[12.5px] text-muted-foreground">
              No prompt versions committed in the last 7 days.
            </p>
          )}
        </Panel>
      </div>
    </div>
  );
}
