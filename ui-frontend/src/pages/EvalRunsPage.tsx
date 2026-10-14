import { ChevronLeft, ChevronRight, RefreshCw, Search, X } from "lucide-react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { GateBadge } from "@/components/evals/AgentCiPanel";
import { PassRateBar } from "@/components/evals/PassRateBar";
import { QualityTrendChart } from "@/components/evals/QualityTrendChart";
import { EVAL_RUNS_PAGE_SIZE, useEvalRuns, useEvalTrend } from "@/hooks/use-evals";
import { caseCounts, costLabel, formatRate, KIND_LABEL } from "@/lib/evalDisplay";
import { formatDurationMs } from "@/lib/format";
import { cn } from "@/lib/utils";
import { Timestamp } from "@/components/shared/Timestamp";
import type { EvalRunFilters } from "@/lib/types";

/** Kind chips: the label, and the `kind` the API filters on. */
const KIND_CHIPS: { label: string; kind: string | null }[] = [
  { label: "All", kind: null },
  { label: "Evals", kind: "evaluate,cli" },
  { label: "Pytest", kind: "pytest" },
  { label: "UI", kind: "ui" },
  { label: "AutoLLM", kind: "optimize" },
  { label: "Dataset checks", kind: "ui-echo" },
];

/** Gate outcomes the API filters on, in display order; `none` = never gated. */
const GATES = ["passed", "failed", "invalid", "none"];

/** The gate select's options; any other valid combination from a URL is added. */
const GATE_OPTIONS: { value: string; label: string }[] = [
  { value: "", label: "Gate: any" },
  { value: "passed", label: "Gate: passed" },
  { value: "failed", label: "Gate: failed" },
  { value: "invalid", label: "Gate: invalid" },
  { value: "failed,invalid", label: "Gate: failed or invalid" },
  { value: "none", label: "Not gated" },
];

/** A comma-separated gate value, cleaned: known outcomes only, deduped, in order. */
function normalizeGate(raw: string | null): string | null {
  const parts = new Set((raw ?? "").split(",").map((s) => s.trim()));
  const kept = GATES.filter((g) => parts.has(g));
  return kept.length ? kept.join(",") : null;
}

export function EvalRunsPage() {
  const navigate = useNavigate();
  // Filters live in the URL so a filtered view can be shared and survives a refresh.
  const [params, setParams] = useSearchParams();
  const kind = params.get("kind") || null;
  const gate = normalizeGate(params.get("gate"));
  const q = params.get("q") ?? "";
  const page = Math.max(1, Number(params.get("page")) || 1);
  const trendAll = params.get("trend") === "all";

  const filters: EvalRunFilters = {
    kind,
    gate,
    q: q || null,
    page,
    page_size: EVAL_RUNS_PAGE_SIZE,
  };
  const runs = useEvalRuns(filters);
  const trend = useEvalTrend(null, trendAll);
  const gateOptions =
    gate && !GATE_OPTIONS.some((o) => o.value === gate)
      ? [...GATE_OPTIONS, { value: gate, label: `Gate: ${gate.split(",").join(" or ")}` }]
      : GATE_OPTIONS;

  /** Write filter changes to the URL; any filter change goes back to page 1. */
  function update(patch: Record<string, string | null>, opts: { keepPage?: boolean } = {}) {
    const next = new URLSearchParams(params);
    for (const [k, v] of Object.entries(patch)) {
      if (v == null || v === "") next.delete(k);
      else next.set(k, v);
    }
    if (!opts.keepPage) next.delete("page");
    setParams(next, { replace: true });
  }

  const rows = runs.data?.rows ?? [];
  const total = runs.data?.total ?? 0;
  const pageSize = runs.data?.page_size ?? EVAL_RUNS_PAGE_SIZE;
  const firstRow = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const lastRow = Math.min(page * pageSize, total);
  const anyFilter = !!(kind || gate || q);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Eval runs"
        description={
          runs.data
            ? `${total.toLocaleString()} run${total === 1 ? "" : "s"}${anyFilter ? " match" : ""}`
            : undefined
        }
      >
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            runs.refetch();
            trend.refetch();
          }}
          disabled={runs.isFetching || trend.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${
              runs.isFetching || trend.isFetching ? "animate-spin" : ""
            }`}
          />
          Refresh
        </Button>
      </PageHeader>

      <Card>
        <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2">
          <CardTitle className="text-sm">Quality trend</CardTitle>
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            <Switch
              checked={trendAll}
              onCheckedChange={(v) => update({ trend: v ? "all" : null }, { keepPage: true })}
              aria-label="Include dataset checks and AutoLLM runs"
              size="sm"
            />
            Include dataset checks and AutoLLM runs
          </label>
        </CardHeader>
        <CardContent>
          {trend.isLoading ? (
            <TableSkeleton rows={3} />
          ) : (
            <QualityTrendChart points={trend.data?.points ?? []} includesAll={trendAll} />
          )}
        </CardContent>
      </Card>

      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1 rounded-md border bg-card p-0.5" role="group" aria-label="Run kind">
          {KIND_CHIPS.map((c) => (
            <button
              key={c.label}
              type="button"
              aria-pressed={kind === c.kind}
              onClick={() => update({ kind: c.kind })}
              className={cn(
                "rounded px-2 py-1 text-xs font-mono font-medium transition-colors",
                kind === c.kind
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
              )}
            >
              {c.label}
            </button>
          ))}
        </div>
        <select
          aria-label="Gate"
          value={gate ?? ""}
          onChange={(e) => update({ gate: e.target.value || null })}
          className="h-9 rounded-md border border-input bg-background px-2 text-sm"
        >
          {gateOptions.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
        <div className="relative">
          <Search className="pointer-events-none absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-muted-foreground" />
          <Input
            type="search"
            aria-label="Search runs"
            placeholder="Run, dataset or agent"
            value={q}
            onChange={(e) => update({ q: e.target.value || null })}
            className="h-9 w-60 pl-7 text-xs"
          />
        </div>
        {anyFilter && (
          <Button variant="ghost" size="sm" onClick={() => update({ kind: null, gate: null, q: null })}>
            <X className="mr-1 h-3.5 w-3.5" />
            Clear
          </Button>
        )}
      </div>

      {runs.isLoading ? (
        <TableSkeleton rows={6} />
      ) : rows.length === 0 ? (
        anyFilter ? (
          <EmptyState
            title="No eval runs match these filters"
            description="Clear the filters to see every run."
          />
        ) : (
          <EmptyState
            title="No eval runs yet"
            description="Call evaluate() — results appear here automatically."
          />
        )
      ) : (
        <div className="space-y-2">
          <div className="rounded-md border bg-card">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead>Run</TableHead>
                  <TableHead>Dataset</TableHead>
                  <TableHead>Agent</TableHead>
                  <TableHead className="w-[200px]" title="Cases passed / (passed + failed). Hover a bar for the checks rate.">
                    Cases
                  </TableHead>
                  <TableHead className="w-[80px]" title="Cases with no verdict">
                    Errored
                  </TableHead>
                  <TableHead className="w-[90px]">Gate</TableHead>
                  <TableHead className="w-[110px]">Branch</TableHead>
                  <TableHead className="w-[100px]">Agent cost</TableHead>
                  <TableHead className="w-[100px]">Avg latency</TableHead>
                  <TableHead className="w-[120px]">Started</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((row) => {
                  const c = caseCounts(row);
                  const cost = costLabel(row.agent_cost_usd, row.agent_cost_known);
                  const meta = row.metadata ?? {};
                  const branch = typeof meta.git_branch === "string" ? meta.git_branch : null;
                  const gateOutcome =
                    row.gate?.outcome ??
                    (typeof meta.gate_outcome === "string" ? meta.gate_outcome : null);
                  return (
                    <TableRow
                      key={row.run_id}
                      className="cursor-pointer"
                      onClick={() => navigate(`/evals/${row.run_id}`)}
                      data-testid={`eval-row-${row.run_id}`}
                    >
                      <TableCell className="font-medium">
                        {row.run_name || row.run_id}
                        {row.kind && (
                          <span className="ml-2 rounded-sm bg-muted px-1 py-px font-mono text-[10px] font-normal text-muted-foreground">
                            {KIND_LABEL[row.kind] ?? row.kind}
                          </span>
                        )}
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {row.dataset_name ?? "—"}
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {row.agent_name ?? "—"}
                      </TableCell>
                      <TableCell>
                        <PassRateBar
                          passRate={row.case_pass_rate}
                          passCount={c.passed}
                          failCount={c.failed}
                          title={`Cases ${formatRate(row.case_pass_rate)} · Checks ${formatRate(row.pass_rate)}`}
                        />
                      </TableCell>
                      <TableCell className="font-mono text-xs tabular-nums">
                        {c.errored > 0 ? c.errored : <span className="text-muted-foreground">0</span>}
                      </TableCell>
                      <TableCell>
                        <GateBadge outcome={gateOutcome} />
                      </TableCell>
                      <TableCell className="max-w-[110px] truncate font-mono text-xs text-muted-foreground" title={branch ?? undefined}>
                        {branch ?? "—"}
                      </TableCell>
                      <TableCell className="font-mono text-xs tabular-nums" title={cost.title}>
                        {cost.text}
                        {cost.partial && <span className="ml-1 text-muted-foreground">*</span>}
                      </TableCell>
                      <TableCell className="font-mono text-xs tabular-nums">
                        {row.avg_latency_ms ? formatDurationMs(row.avg_latency_ms) : "—"}
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        <Timestamp iso={row.started_at} />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>

          <div className="flex items-center justify-end gap-2 text-xs text-muted-foreground">
            <span className="font-mono tabular-nums" data-testid="eval-pager-range">
              {firstRow.toLocaleString()}–{lastRow.toLocaleString()} of {total.toLocaleString()}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => update({ page: page - 1 > 1 ? String(page - 1) : null }, { keepPage: true })}
              disabled={page <= 1 || runs.isFetching}
            >
              <ChevronLeft className="mr-1 h-3.5 w-3.5" />
              Prev
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => update({ page: String(page + 1) }, { keepPage: true })}
              disabled={lastRow >= total || runs.isFetching}
            >
              Next
              <ChevronRight className="ml-1 h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
