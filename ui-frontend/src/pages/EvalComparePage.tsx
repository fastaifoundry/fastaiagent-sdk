import { useMemo } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { ArrowLeftRight, ChevronLeft, RefreshCw, TrendingUp, TrendingDown } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { StatCard } from "@/components/shared/StatCard";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { CaseDiffCard } from "@/components/evals/CaseDiffCard";
import { useEvalCompare, useEvalRuns } from "@/hooks/use-evals";
import { formatCost, formatTimeAgo } from "@/lib/format";
import { cn } from "@/lib/utils";

export function EvalComparePage() {
  const [params, setParams] = useSearchParams();
  const a = params.get("a") ?? undefined;
  const b = params.get("b") ?? undefined;
  const runs = useEvalRuns({ page: 1, page_size: 100 });
  const compare = useEvalCompare(a, b);

  const allRuns = useMemo(() => runs.data?.rows ?? [], [runs.data]);

  function pickA(v: string) {
    const next = new URLSearchParams(params);
    next.set("a", v);
    setParams(next);
  }
  function pickB(v: string) {
    const next = new URLSearchParams(params);
    next.set("b", v);
    setParams(next);
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="Compare eval runs"
        description="Pick two runs of the same dataset to see regressed and improved cases side-by-side."
      >
        <Link to="/evals">
          <Button variant="ghost" size="sm">
            <ChevronLeft className="mr-1.5 h-3.5 w-3.5" />
            Back
          </Button>
        </Link>
        <Button
          variant="outline"
          size="sm"
          onClick={() => compare.refetch()}
          disabled={!a || !b || compare.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${compare.isFetching ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </PageHeader>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Runs</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3 sm:flex-row sm:items-center">
          <div className="flex-1">
            <label className="mb-1 block text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
              Run A (before)
            </label>
            <Select value={a ?? ""} onValueChange={pickA}>
              <SelectTrigger className="w-full text-xs">
                <SelectValue placeholder="Pick run A…" />
              </SelectTrigger>
              <SelectContent>
                {allRuns.map((r) => (
                  <SelectItem key={r.run_id} value={r.run_id}>
                    {r.run_name || r.run_id.slice(0, 12)} ·{" "}
                    {r.dataset_name ?? "—"} ·{" "}
                    {Math.round((r.pass_rate ?? 0) * 100)}% ·{" "}
                    {formatTimeAgo(r.started_at)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <ArrowLeftRight className="mt-5 h-4 w-4 shrink-0 text-muted-foreground" />
          <div className="flex-1">
            <label className="mb-1 block text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
              Run B (after)
            </label>
            <Select value={b ?? ""} onValueChange={pickB}>
              <SelectTrigger className="w-full text-xs">
                <SelectValue placeholder="Pick run B…" />
              </SelectTrigger>
              <SelectContent>
                {allRuns.map((r) => (
                  <SelectItem key={r.run_id} value={r.run_id}>
                    {r.run_name || r.run_id.slice(0, 12)} ·{" "}
                    {r.dataset_name ?? "—"} ·{" "}
                    {Math.round((r.pass_rate ?? 0) * 100)}% ·{" "}
                    {formatTimeAgo(r.started_at)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {!a || !b ? (
        <EmptyState
          title="Pick two runs"
          description="Both selectors above must be set. Datasets can differ — cases are matched by ordinal and input."
        />
      ) : compare.isLoading ? (
        <TableSkeleton rows={5} />
      ) : compare.error ? (
        <EmptyState
          title="Compare failed"
          description={
            compare.error instanceof Error
              ? compare.error.message
              : "Couldn't load comparison."
          }
        />
      ) : !compare.data ? null : (
        <CompareResults data={compare.data} />
      )}
    </div>
  );
}

function CompareResults({
  data,
}: {
  data: NonNullable<ReturnType<typeof useEvalCompare>["data"]>;
}) {
  const {
    run_a,
    run_b,
    regressed,
    improved,
    unchanged_pass,
    unchanged_fail,
    pass_rate_delta,
    cost_delta_usd,
  } = data;
  const passRateDeltaPct = Math.round(pass_rate_delta * 100);
  const improvedPill = pass_rate_delta > 0;
  const regressedPill = pass_rate_delta < 0;

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatCard
          label="Pass-rate Δ"
          value={`${passRateDeltaPct >= 0 ? "+" : ""}${passRateDeltaPct}%`}
        />
        <StatCard
          label="Cost Δ"
          value={`${cost_delta_usd >= 0 ? "+" : ""}${formatCost(cost_delta_usd)}`}
        />
        <StatCard label="Regressed" value={String(regressed.length)} />
        <StatCard label="Improved" value={String(improved.length)} />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <RunSummary label="A (before)" run={run_a} />
        <RunSummary label="B (after)" run={run_b} />
      </div>

      <p className="text-xs text-muted-foreground">
        {unchanged_pass} case{unchanged_pass === 1 ? "" : "s"} pass → pass,{" "}
        {unchanged_fail} fail → fail. Only regressions and improvements expand
        below.
        {improvedPill && (
          <span className="ml-2 inline-flex items-center gap-1 text-fa-success">
            <TrendingUp className="h-3 w-3" /> net improvement
          </span>
        )}
        {regressedPill && (
          <span className="ml-2 inline-flex items-center gap-1 text-destructive">
            <TrendingDown className="h-3 w-3" /> net regression
          </span>
        )}
      </p>

      {regressed.length > 0 && (
        <Card className="border-destructive/30">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <TrendingDown className="h-3.5 w-3.5 text-destructive" />
              Regressed — passed in A, failed in B ({regressed.length})
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {regressed.map((pair, i) => (
              <CaseDiffCard
                key={`r-${i}`}
                case_={pair.b}
                deltas={pair.scorer_deltas}
                otherActual={pair.a.actual_output}
                thisLabel="B"
                otherLabel="A"
              />
            ))}
          </CardContent>
        </Card>
      )}

      {improved.length > 0 && (
        <Card className="border-fa-success/30">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <TrendingUp className="h-3.5 w-3.5 text-fa-success" />
              Improved — failed in A, passed in B ({improved.length})
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {improved.map((pair, i) => (
              <CaseDiffCard
                key={`i-${i}`}
                case_={pair.b}
                deltas={pair.scorer_deltas}
                otherActual={pair.a.actual_output}
                thisLabel="B"
                otherLabel="A"
              />
            ))}
          </CardContent>
        </Card>
      )}

      {regressed.length === 0 && improved.length === 0 && (
        <EmptyState
          title="No case-level differences"
          description="Every case matched by ordinal had the same pass/fail outcome. Scorer scores may still have drifted — open each run separately to inspect."
        />
      )}
    </div>
  );
}

function RunSummary({
  label,
  run,
}: {
  label: string;
  run: NonNullable<ReturnType<typeof useEvalCompare>["data"]>["run_a"];
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between text-xs">
          <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
            {label}
          </span>
          <Link
            to={`/evals/${run.run_id}`}
            className="font-mono text-xs hover:text-primary"
          >
            {run.run_name || run.run_id.slice(0, 12)}
          </Link>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-1 text-xs">
        <Row k="Dataset" v={run.dataset_name ?? "—"} />
        <Row k="Agent" v={`${run.agent_name ?? "—"} · ${run.agent_version ?? ""}`} />
        <Row
          k="Pass rate"
          v={`${Math.round((run.pass_rate ?? 0) * 100)}% (${run.pass_count}/${(run.pass_count ?? 0) + (run.fail_count ?? 0)})`}
        />
        <Row k="Cost" v={formatCost(run.cost_usd ?? 0)} />
        <Row k="Started" v={formatTimeAgo(run.started_at ?? "")} />
      </CardContent>
    </Card>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className={cn("flex items-center justify-between")}>
      <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        {k}
      </span>
      <span className="font-mono">{v}</span>
    </div>
  );
}
