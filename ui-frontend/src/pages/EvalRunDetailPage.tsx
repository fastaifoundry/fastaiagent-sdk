import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  Play,
  RefreshCw,
  Search,
} from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { StatCard } from "@/components/shared/StatCard";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { CaseDiffCard } from "@/components/evals/CaseDiffCard";
import { PassRateBar } from "@/components/evals/PassRateBar";
import { useEvalRun } from "@/hooks/use-evals";
import { formatCost, formatDurationMs, formatTimeAgo } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { EvalCaseFilters, EvalCaseRow } from "@/lib/types";

function stringifyShort(value: unknown, max = 120): string {
  if (value == null) return "—";
  if (typeof value === "string") {
    return value.length > max ? `${value.slice(0, max)}…` : value;
  }
  try {
    const s = JSON.stringify(value);
    return s.length > max ? `${s.slice(0, max)}…` : s;
  } catch {
    return String(value);
  }
}

export function EvalRunDetailPage() {
  const { runId } = useParams<{ runId: string }>();
  const [filters, setFilters] = useState<EvalCaseFilters>({});
  const [expanded, setExpanded] = useState<string | null>(null);
  const query = useEvalRun(runId, filters);

  const scorerNames = useMemo(() => {
    if (!query.data) return [];
    const summary = query.data.run.scorer_summary ?? {};
    const fromSummary = Object.keys(summary);
    if (fromSummary.length) return fromSummary;
    return Array.from(
      new Set(
        (query.data.cases ?? []).flatMap((c) =>
          Object.keys(c.per_scorer ?? {})
        )
      )
    );
  }, [query.data]);

  if (query.isLoading && !query.data) return <TableSkeleton rows={8} />;
  if (query.error || !query.data) {
    return (
      <EmptyState
        title="Eval run not found"
        description="This run id doesn't exist in the local DB."
      />
    );
  }

  const { run, cases, total_cases } = query.data;

  return (
    <div className="space-y-5">
      <PageHeader
        title={run.run_name || `Run ${run.run_id.slice(0, 8)}`}
        description={`Dataset ${run.dataset_name ?? "—"} · Agent ${run.agent_name ?? "—"}`}
      >
        <Link to="/evals">
          <Button variant="ghost" size="sm">
            <ChevronLeft className="mr-1.5 h-3.5 w-3.5" />
            Back
          </Button>
        </Link>
        <Link to={`/evals/compare?a=${run.run_id}`}>
          <Button variant="outline" size="sm">
            Compare with…
          </Button>
        </Link>
        <Button
          variant="outline"
          size="sm"
          onClick={() => query.refetch()}
          disabled={query.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${query.isFetching ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </PageHeader>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-6">
        <StatCard label="Pass rate" value={`${Math.round((run.pass_rate ?? 0) * 100)}%`} />
        <StatCard label="Passed" value={String(run.pass_count ?? 0)} />
        <StatCard label="Failed" value={String(run.fail_count ?? 0)} />
        <StatCard
          label="Avg latency"
          value={formatDurationMs(run.avg_latency_ms ?? 0)}
        />
        <StatCard label="Total cost" value={formatCost(run.cost_usd ?? 0)} />
        <StatCard
          label="Started"
          value={formatTimeAgo(run.started_at ?? "")}
        />
      </div>

      {run.scorer_summary && Object.keys(run.scorer_summary).length > 0 && (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
            Scorers
          </span>
          {Object.entries(run.scorer_summary).map(([scorer, counts]) => {
            const total = counts.pass + counts.fail;
            const rate = total > 0 ? counts.pass / total : 0;
            return (
              <span
                key={scorer}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 font-mono",
                  rate >= 0.9
                    ? "border-fa-success/50 text-fa-success"
                    : rate >= 0.7
                    ? "border-fa-warning/50 text-fa-warning"
                    : "border-destructive/50 text-destructive"
                )}
                title={`${counts.pass}/${total} passed`}
              >
                {scorer}
                <span className="tabular-nums opacity-80">
                  {counts.pass}/{total}
                </span>
              </span>
            );
          })}
        </div>
      )}

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <CardTitle className="text-sm">
            Cases
            <span className="ml-2 text-xs font-normal text-muted-foreground">
              {cases.length} of {total_cases} shown
            </span>
          </CardTitle>
          <div className="flex flex-wrap items-center gap-2">
            <div className="relative">
              <Search className="pointer-events-none absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-muted-foreground" />
              <Input
                type="search"
                placeholder="Search input / expected / actual"
                value={filters.q ?? ""}
                onChange={(e) =>
                  setFilters((f) => ({ ...f, q: e.target.value || undefined }))
                }
                className="h-8 w-64 pl-7 text-xs"
              />
            </div>
            <Select
              value={filters.outcome ?? "all"}
              onValueChange={(v) =>
                setFilters((f) => ({
                  ...f,
                  outcome: v === "all" ? null : (v as "passed" | "failed"),
                }))
              }
            >
              <SelectTrigger className="h-8 w-[130px] text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All outcomes</SelectItem>
                <SelectItem value="passed">Passed only</SelectItem>
                <SelectItem value="failed">Failed only</SelectItem>
              </SelectContent>
            </Select>
            <Select
              value={filters.scorer ?? "all"}
              onValueChange={(v) =>
                setFilters((f) => ({
                  ...f,
                  scorer: v === "all" ? null : v,
                }))
              }
            >
              <SelectTrigger className="h-8 w-[150px] text-xs">
                <SelectValue placeholder="Scorer" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All scorers</SelectItem>
                {scorerNames.map((s) => (
                  <SelectItem key={s} value={s}>
                    {s}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {cases.length === 0 ? (
            <div className="px-6 py-10">
              <EmptyState
                title="No cases match these filters"
                description="Clear the filter bar to see every case."
              />
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead className="w-[30px]" />
                  <TableHead className="w-[50px]">#</TableHead>
                  <TableHead>Input</TableHead>
                  <TableHead>Expected</TableHead>
                  <TableHead>Actual</TableHead>
                  {scorerNames.map((name) => (
                    <TableHead key={name} className="w-[120px]">
                      {name}
                    </TableHead>
                  ))}
                  <TableHead className="w-[80px]" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {cases.map((c) => (
                  <CaseRow
                    key={c.case_id}
                    case_={c}
                    scorerNames={scorerNames}
                    expanded={expanded === c.case_id}
                    onToggle={() =>
                      setExpanded((cur) => (cur === c.case_id ? null : c.case_id))
                    }
                  />
                ))}
              </TableBody>
            </Table>
          )}
          {cases.length > 0 && (
            <div className="border-t p-3">
              <PassRateBar
                passRate={run.pass_rate}
                passCount={run.pass_count}
                failCount={run.fail_count}
              />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function CaseRow({
  case_,
  scorerNames,
  expanded,
  onToggle,
}: {
  case_: EvalCaseRow;
  scorerNames: string[];
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <>
      <TableRow>
        <TableCell>
          <button
            type="button"
            aria-label={expanded ? "Collapse case" : "Expand case"}
            onClick={onToggle}
            className="inline-flex h-5 w-5 items-center justify-center rounded hover:bg-muted"
          >
            {expanded ? (
              <ChevronDown className="h-3 w-3" />
            ) : (
              <ChevronRight className="h-3 w-3" />
            )}
          </button>
        </TableCell>
        <TableCell className="font-mono text-xs text-muted-foreground">
          {case_.ordinal}
        </TableCell>
        <TableCell
          className="max-w-[200px] truncate font-mono text-xs"
          title={stringifyShort(case_.input, 2000)}
        >
          {stringifyShort(case_.input)}
        </TableCell>
        <TableCell
          className="max-w-[200px] truncate font-mono text-xs"
          title={stringifyShort(case_.expected_output, 2000)}
        >
          {stringifyShort(case_.expected_output)}
        </TableCell>
        <TableCell
          className="max-w-[200px] truncate font-mono text-xs"
          title={stringifyShort(case_.actual_output, 2000)}
        >
          {stringifyShort(case_.actual_output)}
        </TableCell>
        {scorerNames.map((name) => {
          const score = case_.per_scorer?.[name];
          if (!score)
            return (
              <TableCell key={name} className="text-xs text-muted-foreground">
                —
              </TableCell>
            );
          return (
            <TableCell key={name}>
              <span
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs font-mono",
                  score.passed
                    ? "bg-fa-success/10 text-fa-success"
                    : "bg-destructive/10 text-destructive"
                )}
              >
                <span className="h-1.5 w-1.5 rounded-full bg-current" />
                {score.passed ? "pass" : "fail"}
                <span className="tabular-nums opacity-70">
                  {(score.score ?? 0).toFixed(2)}
                </span>
              </span>
            </TableCell>
          );
        })}
        <TableCell>
          <div className="flex items-center gap-1">
            {case_.trace_id && (
              <Link
                to={`/traces/${case_.trace_id}`}
                title="Open trace"
                className="inline-flex items-center text-muted-foreground hover:text-primary"
              >
                <ExternalLink className="h-3.5 w-3.5" />
              </Link>
            )}
            {case_.trace_id && (
              <Link
                to={`/traces/${case_.trace_id}/replay`}
                title="Open in Replay"
                className="ml-1 inline-flex items-center text-muted-foreground hover:text-primary"
              >
                <Play className="h-3.5 w-3.5" />
              </Link>
            )}
          </div>
        </TableCell>
      </TableRow>
      {expanded && (
        <TableRow className="hover:bg-transparent">
          <TableCell
            colSpan={6 + scorerNames.length}
            className="bg-muted/30 p-4"
          >
            <CaseDiffCard case_={case_} />
          </TableCell>
        </TableRow>
      )}
    </>
  );
}
