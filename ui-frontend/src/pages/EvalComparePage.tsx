import { useMemo, useState, type ReactNode } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  AlertTriangle,
  ArrowLeftRight,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  RefreshCw,
  TrendingUp,
  TrendingDown,
} from "lucide-react";
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
import { formatDateTime } from "@/lib/format";
import {
  caseCounts,
  caseValueText,
  costLabel,
  formatCostDelta,
  formatRate,
  formatRateDelta,
} from "@/lib/evalDisplay";
import { cn } from "@/lib/utils";
import { Timestamp } from "@/components/shared/Timestamp";
import type {
  EvalCaseRow,
  EvalCompareErroredPair,
  EvalComparePair,
  EvalCompareResponse,
  EvalRunRow,
} from "@/lib/types";

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
                  <RunOption key={r.run_id} run={r} />
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
                  <RunOption key={r.run_id} run={r} />
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

function RunOption({ run }: { run: EvalRunRow }) {
  return (
    <SelectItem value={run.run_id}>
      {run.run_name || run.run_id.slice(0, 12)} · {run.dataset_name ?? "—"} · cases{" "}
      {formatRate(run.case_pass_rate)} · <Timestamp iso={run.started_at} />
    </SelectItem>
  );
}

/** A section that starts collapsed; its title carries the count. */
function Collapsible({
  title,
  count,
  testId,
  children,
}: {
  title: string;
  count: number;
  testId: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  if (count === 0) return null;
  return (
    <Card data-testid={testId}>
      <CardHeader className="py-3">
        <button
          type="button"
          aria-expanded={open}
          onClick={() => setOpen(!open)}
          className="flex items-center gap-2 text-left text-sm font-medium"
        >
          {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          {title} ({count})
        </button>
      </CardHeader>
      {open && <CardContent className="space-y-2 text-xs">{children}</CardContent>}
    </Card>
  );
}

function CaseLine({ case_, extra }: { case_: EvalCaseRow; extra?: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-3 rounded-md border px-2 py-1.5">
      <div className="min-w-0">
        <span className="mr-2 font-mono text-muted-foreground">#{case_.ordinal}</span>
        <span className="font-mono">{caseValueText(case_.input, 160)}</span>
        {extra}
      </div>
      {case_.trace_id && (
        <Link
          to={`/traces/${case_.trace_id}`}
          className="shrink-0 text-muted-foreground hover:text-primary"
        >
          trace
        </Link>
      )}
    </div>
  );
}

const SIDE_TEXT: Record<EvalCompareErroredPair["side"], string> = {
  a: "errored in A (before)",
  b: "errored in B (after)",
  both: "errored in both",
};

function ErroredLine({ pair }: { pair: EvalCompareErroredPair }) {
  const errors = [
    pair.side !== "b" ? pair.a.error : null,
    pair.side !== "a" ? pair.b.error : null,
  ].filter(Boolean);
  return (
    <CaseLine
      case_={pair.b}
      extra={
        <div className="mt-0.5 text-muted-foreground">
          <span className="mr-2 rounded-sm bg-muted px-1 py-px font-mono">{SIDE_TEXT[pair.side]}</span>
          {errors.length > 0 ? errors.join(" · ") : "a check could not run, or none applied"}
        </div>
      }
    />
  );
}

function DriftLine({ pair }: { pair: EvalComparePair }) {
  return (
    <CaseLine
      case_={pair.b}
      extra={
        <div className="mt-0.5 flex flex-wrap gap-x-3 font-mono text-muted-foreground">
          {pair.scorer_deltas.map((d) => (
            <span key={d.scorer}>
              {d.scorer}: {d.score_before != null ? d.score_before.toFixed(2) : "—"} →{" "}
              {d.score_after != null ? d.score_after.toFixed(2) : "—"}
              {d.score_delta != null && (
                <span className={d.score_delta < 0 ? "text-destructive" : "text-fa-success"}>
                  {" "}
                  ({d.score_delta > 0 ? "+" : ""}
                  {d.score_delta.toFixed(2)})
                </span>
              )}
            </span>
          ))}
        </div>
      }
    />
  );
}

function CompareResults({ data }: { data: EvalCompareResponse }) {
  const {
    run_a,
    run_b,
    regressed,
    improved,
    unchanged_pass,
    unchanged_fail,
    pass_rate_delta,
  } = data;
  const errored = data.errored ?? [];
  const onlyA = data.only_in_a ?? [];
  const onlyB = data.only_in_b ?? [];
  const drifted = data.drifted ?? [];
  const comparable = data.pass_rate_comparable !== false;
  const casesDelta = data.case_pass_rate_delta ?? null;
  // The headline delta is the cases rate; fall back to checks only if absent.
  const headlineDelta = casesDelta ?? pass_rate_delta ?? null;
  const improvedPill = headlineDelta != null && headlineDelta > 0;
  const regressedPill = headlineDelta != null && headlineDelta < 0;

  return (
    <div className="space-y-5">
      {!comparable && (
        <div
          className="flex items-start gap-2 rounded-md border border-fa-warning/40 bg-fa-warning/10 px-3 py-2 text-xs text-fa-warning"
          data-testid="not-comparable-warning"
        >
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <div>
            One run was stored before 1.89, when the checks rate counted checks that could not run
            or did not apply as failures. The checks Δ is not like for like; the cases Δ is.
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
        <StatCard
          label="Cases Δ"
          value={formatRateDelta(casesDelta)}
          title="Change in passed cases / (passed + failed cases)"
        />
        <StatCard
          label="Checks Δ"
          value={formatRateDelta(pass_rate_delta)}
          valueTitle={pass_rate_delta == null ? "a run has no checks rate" : undefined}
          hint={comparable ? undefined : "not like for like"}
          title="Change in overall.pass_rate, what --eval-fail-under gates"
        />
        <StatCard
          label="Agent cost Δ"
          value={formatCostDelta(data.agent_cost_delta_usd)}
          valueTitle={data.agent_cost_delta_usd == null ? "a run was not traced" : undefined}
        />
        <StatCard
          label="Eval cost Δ"
          value={formatCostDelta(data.eval_cost_delta_usd)}
          valueTitle={data.eval_cost_delta_usd == null ? "a run's scoring was not traced" : undefined}
        />
        <StatCard label="Regressed" value={String(regressed.length)} />
        <StatCard label="Improved" value={String(improved.length)} />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <RunSummary label="A (before)" run={run_a} />
        <RunSummary label="B (after)" run={run_b} />
      </div>

      <p className="text-xs text-muted-foreground" data-testid="compare-summary">
        {unchanged_pass} case{unchanged_pass === 1 ? "" : "s"} pass → pass,{" "}
        {unchanged_fail} fail → fail
        {errored.length > 0 && `, ${errored.length} errored`}
        {onlyA.length + onlyB.length > 0 && `, ${onlyA.length + onlyB.length} unmatched`}
        {drifted.length > 0 && `, ${drifted.length} with score drift`}.
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
                otherCase={pair.a}
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
                otherCase={pair.a}
                thisLabel="B"
                otherLabel="A"
              />
            ))}
          </CardContent>
        </Card>
      )}

      {regressed.length === 0 && improved.length === 0 && (
        <EmptyState
          title="No case changed between passed and failed"
          description={
            "Every matched case kept its outcome." +
            (errored.length || onlyA.length || onlyB.length || drifted.length
              ? " Errored, unmatched and score-drift cases are listed below."
              : " No case errored, every case matched, and no score moved by 0.1 or more.")
          }
        />
      )}

      <Collapsible title="Errored — not compared" count={errored.length} testId="compare-errored">
        <p className="text-muted-foreground">
          A case with no verdict on either side is neither a regression nor an improvement.
        </p>
        {errored.map((pair, i) => (
          <ErroredLine key={`e-${i}`} pair={pair} />
        ))}
      </Collapsible>

      <Collapsible
        title="Unmatched — in one run only"
        count={onlyA.length + onlyB.length}
        testId="compare-unmatched"
      >
        {onlyA.length > 0 && (
          <div className="space-y-1.5">
            <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
              Only in A (baseline) · {onlyA.length}
            </div>
            {onlyA.map((c) => (
              <CaseLine key={c.case_id} case_={c} />
            ))}
          </div>
        )}
        {onlyB.length > 0 && (
          <div className="space-y-1.5">
            <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
              Only in B (current) · {onlyB.length}
            </div>
            {onlyB.map((c) => (
              <CaseLine key={c.case_id} case_={c} />
            ))}
          </div>
        )}
      </Collapsible>

      <Collapsible title="Score drift — same outcome, score moved" count={drifted.length} testId="compare-drift">
        {drifted.map((pair, i) => (
          <DriftLine key={`d-${i}`} pair={pair} />
        ))}
      </Collapsible>
    </div>
  );
}

function RunSummary({ label, run }: { label: string; run: EvalRunRow }) {
  const c = caseCounts(run);
  const agent = costLabel(run.agent_cost_usd, run.agent_cost_known);
  const evalCost = costLabel(run.eval_cost_usd, run.eval_cost_known);
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
        <Row k="Cases" v={`${formatRate(run.case_pass_rate)} (${c.passed}/${c.judged})`} />
        <Row
          k="Checks"
          v={`${formatRate(run.pass_rate)} (${run.pass_count ?? 0}/${(run.pass_count ?? 0) + (run.fail_count ?? 0)})`}
        />
        <Row k="Errored" v={String(c.errored)} />
        <Row k="Agent cost" v={agent.text + (agent.partial ? " (partial)" : "")} />
        <Row k="Eval cost" v={evalCost.text + (evalCost.partial ? " (partial)" : "")} />
        <Row k="Started" v={formatDateTime(run.started_at ?? "")} />
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
