import { useMemo, useState, type ReactNode } from "react";
import { Link, useParams } from "react-router-dom";
import {
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  Info,
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
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { AgentCiPanel } from "@/components/evals/AgentCiPanel";
import { CaseDiffCard } from "@/components/evals/CaseDiffCard";
import { EvalRunHeadline } from "@/components/evals/EvalRunHeadline";
import { HardeningCard } from "@/components/evals/HardeningCard";
import { PassRateBar } from "@/components/evals/PassRateBar";
import { ScorerConfigCard } from "@/components/evals/ScorerConfigCard";
import { VerdictBadge } from "@/components/evals/VerdictTable";
import { useEvalRun } from "@/hooks/use-evals";
import { caseCounts, caseValueText, formatRate, isEchoRun } from "@/lib/evalDisplay";
import { cn } from "@/lib/utils";
import { evalCaseOutcome, evalVerdictOutcome } from "@/lib/types";
import type { EvalCaseFilters, EvalCaseOutcome, EvalCaseRow, EvalRunRow } from "@/lib/types";

const CASE_OUTCOME_CHIP: Record<EvalCaseOutcome, string> = {
  passed: "bg-fa-success/10 text-fa-success",
  failed: "bg-destructive/10 text-destructive",
  // Neutral: an errored case says nothing about the agent's quality.
  errored: "bg-muted text-muted-foreground",
};

function Banner({ children, testId }: { children: ReactNode; testId: string }) {
  return (
    <div
      className="flex items-start gap-2 rounded-md border bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
      data-testid={testId}
    >
      <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
      <div>{children}</div>
    </div>
  );
}

function ScorerChips({ summary }: { summary: NonNullable<EvalRunRow["scorer_summary"]> }) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs" data-testid="scorer-chips">
      <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        Scorers
      </span>
      {Object.entries(summary).map(([scorer, counts]) => {
        const judged = counts.pass + counts.fail;
        const rate = judged > 0 ? counts.pass / judged : null;
        const extra = [
          counts.errored ? `${counts.errored} errored` : null,
          counts.skipped ? `${counts.skipped} n/a` : null,
        ].filter(Boolean);
        return (
          <span
            key={scorer}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 font-mono",
              rate == null
                ? "border-border text-muted-foreground"
                : rate >= 0.9
                ? "border-fa-success/50 text-fa-success"
                : rate >= 0.7
                ? "border-fa-warning/50 text-fa-warning"
                : "border-destructive/50 text-destructive"
            )}
            title={[`${counts.pass}/${judged} passed`, ...extra].join(" · ")}
          >
            {scorer}
            <span className="tabular-nums opacity-80">
              {counts.pass}/{judged}
            </span>
            {extra.length > 0 && (
              <span className="tabular-nums text-muted-foreground">· {extra.join(" · ")}</span>
            )}
          </span>
        );
      })}
    </div>
  );
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
  // From the run row, not the (filtered) case list on screen.
  const counts = caseCounts(run);
  const meta = run.metadata ?? {};

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

      {isEchoRun(run) && (
        <Banner testId="echo-banner">
          <strong className="font-medium text-foreground">
            Dataset check — not a model evaluation.
          </strong>{" "}
          No agent ran: each input was echoed back, so a case passes only where its expected
          output equals its input.
        </Banner>
      )}
      {run.summary_version == null && (
        <Banner testId="legacy-banner">
          This run was stored before 1.89. Its checks numbers follow the old rules, which counted
          checks that could not run or did not apply as failures. The case numbers are recomputed
          from its cases.
        </Banner>
      )}

      <EvalRunHeadline run={run} />

      {run.scorer_summary && Object.keys(run.scorer_summary).length > 0 && (
        <ScorerChips summary={run.scorer_summary} />
      )}

      {run.gate && <AgentCiPanel gate={run.gate} runId={run.run_id} />}

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <CardTitle className="text-sm">
            Cases
            <span className="ml-2 text-xs font-normal text-muted-foreground">
              {cases.length} of {total_cases} shown
            </span>
            {counts.errored > 0 && (
              <span
                className="ml-2 rounded-md bg-muted px-2 py-0.5 text-xs font-normal text-muted-foreground"
                title="No verdict: the agent failed, a check could not run, or no check applied. Left out of the cases rate."
              >
                {counts.errored} errored
              </span>
            )}
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
                  outcome: v === "all" ? null : (v as EvalCaseOutcome),
                }))
              }
            >
              <SelectTrigger className="h-8 w-[140px] text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All outcomes</SelectItem>
                <SelectItem value="passed">Passed only</SelectItem>
                <SelectItem value="failed">Failed only</SelectItem>
                <SelectItem value="errored">Errored only</SelectItem>
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
                  <TableHead className="w-[90px]">Case</TableHead>
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
                passRate={run.case_pass_rate}
                passCount={counts.passed}
                failCount={counts.failed}
                title={`Cases ${formatRate(run.case_pass_rate)} · Checks ${formatRate(run.pass_rate)}`}
              />
            </div>
          )}
        </CardContent>
      </Card>

      <ScorerConfigCard config={meta.scorer_config} />
      <HardeningCard hardening={meta.hardening} />
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
  const outcome = evalCaseOutcome(case_);
  // The agent itself failed: no scorer ran.
  const agentErrored = !!case_.error;
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
        <TableCell>
          <span
            className={cn(
              "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 font-mono text-xs",
              CASE_OUTCOME_CHIP[outcome]
            )}
            data-testid={`case-outcome-${case_.case_id}`}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-current" />
            {outcome}
          </span>
        </TableCell>
        <TableCell
          className="max-w-[200px] truncate font-mono text-xs"
          title={caseValueText(case_.input, 2000)}
        >
          {caseValueText(case_.input)}
        </TableCell>
        <TableCell
          className="max-w-[200px] truncate font-mono text-xs"
          title={caseValueText(case_.expected_output, 2000)}
        >
          {caseValueText(case_.expected_output)}
        </TableCell>
        <TableCell
          className="max-w-[200px] truncate font-mono text-xs"
          title={caseValueText(case_.actual_output, 2000)}
        >
          {caseValueText(case_.actual_output)}
        </TableCell>
        {agentErrored && (
          <TableCell
            colSpan={scorerNames.length || 1}
            className="text-xs"
            title={case_.error ?? undefined}
          >
            {/* Neutral, not destructive: an infra failure is NOT an agent-quality
                miss, and colouring it red would read as one. */}
            <span className="truncate align-middle text-muted-foreground/80">
              {case_.error}
            </span>
          </TableCell>
        )}
        {!agentErrored && scorerNames.map((name) => {
          const verdict = case_.per_scorer?.[name];
          if (!verdict)
            return (
              <TableCell key={name} className="text-xs text-muted-foreground">
                —
              </TableCell>
            );
          const vo = evalVerdictOutcome(verdict);
          const showScore = vo === "passed" || vo === "failed";
          return (
            <TableCell key={name} title={verdict.reason ?? undefined}>
              <span className="inline-flex items-center gap-1.5">
                <VerdictBadge outcome={vo} />
                {showScore && typeof verdict.score === "number" && (
                  <span className="font-mono text-xs tabular-nums text-muted-foreground">
                    {verdict.score.toFixed(2)}
                  </span>
                )}
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
            colSpan={7 + scorerNames.length}
            className="bg-muted/30 p-4"
          >
            <CaseDiffCard case_={case_} />
          </TableCell>
        </TableRow>
      )}
    </>
  );
}
