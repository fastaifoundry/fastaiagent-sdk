import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ChevronDown, ChevronLeft, ChevronRight, RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatCard } from "@/components/shared/StatCard";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { SimulationTranscriptView } from "@/components/simulations/SimulationTranscriptView";
import { useSimulationRun } from "@/hooks/use-simulations";
import { formatDateTime } from "@/lib/format";
import { formatRate } from "@/lib/evalDisplay";
import { cn } from "@/lib/utils";
import { HardeningCard } from "@/components/evals/HardeningCard";
import type {
  CriterionVerdict,
  EvalCaseOutcome,
  EvalHardening,
  SimulationCase,
  SimulationRunRow,
} from "@/lib/types";

/**
 * Passed / failed among judged scenarios, and errored (not finished or not
 * judged). Not read from `fail_count`: it includes errored scenarios on rows
 * written before 1.89.0 and excludes them after.
 */
function simCounts(run: SimulationRunRow) {
  const passed = run.pass_count ?? 0;
  const errored = run.errored_count ?? 0;
  const judged = run.judged_count ?? Math.max(0, (run.scenario_count ?? 0) - errored);
  return { passed, errored, failed: Math.max(0, judged - passed), judged };
}

export function SimulationDetailPage() {
  const { runId } = useParams<{ runId: string }>();
  const [expanded, setExpanded] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<EvalCaseOutcome | null>(null);
  const query = useSimulationRun(runId, { outcome });

  if (query.isLoading && !query.data) return <TableSkeleton rows={8} />;
  if (query.error || !query.data) {
    return (
      <EmptyState
        title="Simulation run not found"
        description="This run id doesn't exist in the local DB."
      />
    );
  }

  const { run, cases, total_cases } = query.data;
  const counts = simCounts(run);

  return (
    <div className="space-y-5">
      <PageHeader
        title={run.run_name || `Run ${run.run_id.slice(0, 8)}`}
        description={`Agent ${run.agent_name ?? "—"} · ${total_cases} scenario${
          total_cases === 1 ? "" : "s"
        }`}
      >
        <Link to="/simulations">
          <Button variant="ghost" size="sm">
            <ChevronLeft className="mr-1.5 h-3.5 w-3.5" />
            Back
          </Button>
        </Link>
        <Button
          variant="outline"
          size="sm"
          onClick={() => query.refetch()}
          disabled={query.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${
              query.isFetching ? "animate-spin" : ""
            }`}
          />
          Refresh
        </Button>
      </PageHeader>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
        <StatCard label="Passed" value={String(counts.passed)} hint={`of ${counts.judged} judged`} />
        <StatCard
          label="Failed"
          value={String(counts.failed)}
          hint="judged"
          title="Scenarios the judge ran on and failed"
        />
        <StatCard
          label="Errored"
          value={String(counts.errored)}
          hint="could not run"
          title="Scenarios that did not finish, or that the judge could not run on"
        />
        <StatCard
          label="Pass rate"
          value={formatRate(run.pass_rate)}
          hint="of all scenarios"
          title="Passed / all scenarios. An errored scenario counts as not passed."
        />
        <StatCard label="Started" value={formatDateTime(run.started_at ?? "")} />
      </div>

      <div className="flex items-center gap-2">
        <select
          aria-label="Outcome"
          value={outcome ?? ""}
          onChange={(e) => setOutcome((e.target.value || null) as EvalCaseOutcome | null)}
          className="h-9 rounded-md border border-input bg-background px-2 text-sm"
        >
          <option value="">All outcomes</option>
          <option value="passed">Passed</option>
          <option value="failed">Failed</option>
          <option value="errored">Errored (could not run)</option>
        </select>
        <span className="text-xs text-muted-foreground">
          {cases.length} of {total_cases} shown
        </span>
      </div>

      <HardeningCard
        hardening={(run.metadata as { hardening?: EvalHardening } | null)?.hardening}
      />

      {cases.length === 0 ? (
        outcome ? (
          <EmptyState title="No scenarios match" description="Pick another outcome to see more." />
        ) : (
          <EmptyState title="No scenarios" description="This run has no cases." />
        )
      ) : (
        <div className="space-y-3">
          {cases.map((c) => (
            <ScenarioCard
              key={c.case_id}
              case_={c}
              expanded={expanded === c.case_id}
              onToggle={() =>
                setExpanded((cur) => (cur === c.case_id ? null : c.case_id))
              }
            />
          ))}
        </div>
      )}
    </div>
  );
}

function CriterionChip({ verdict }: { verdict: CriterionVerdict }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs",
        verdict.errored
          ? "bg-muted text-muted-foreground"
          : verdict.passed
          ? "bg-fa-success/10 text-fa-success"
          : "bg-destructive/10 text-destructive"
      )}
      title={verdict.reason ?? undefined}
      data-errored={verdict.errored ? "true" : undefined}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      <span className="font-mono opacity-70">{verdict.kind}</span>
      {verdict.criterion}
      {verdict.errored && <span className="font-mono">· could not run</span>}
    </span>
  );
}

function ScenarioCard({
  case_,
  expanded,
  onToggle,
}: {
  case_: SimulationCase;
  expanded: boolean;
  onToggle: () => void;
}) {
  const errored = !!case_.error;
  const passed = !errored && case_.passed === 1;
  return (
    <Card>
      <CardHeader
        className="cursor-pointer flex-row items-center justify-between gap-3 py-3"
        onClick={onToggle}
      >
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            {expanded ? (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            )}
            <CardTitle className="text-sm">
              {case_.scenario_name ?? `Scenario ${case_.ordinal}`}
            </CardTitle>
            <span
              className={cn(
                "rounded-md px-2 py-0.5 text-xs font-mono",
                errored
                  ? "bg-muted text-muted-foreground"
                  : passed
                  ? "bg-fa-success/10 text-fa-success"
                  : "bg-destructive/10 text-destructive"
              )}
              data-testid={`scenario-badge-${case_.case_id}`}
            >
              {errored ? "ERRORED" : passed ? "PASS" : "FAIL"}
            </span>
          </div>
          {errored && (
            <p className="ml-6 mt-1 truncate text-xs text-muted-foreground" title={case_.error ?? ""}>
              {case_.error}
            </p>
          )}
        </div>
        <span className="shrink-0 text-xs text-muted-foreground">
          {case_.transcript.length} turns
        </span>
      </CardHeader>
      {expanded && (
        <CardContent className="space-y-4">
          {case_.per_criterion.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {case_.per_criterion.map((v, i) => (
                <CriterionChip key={i} verdict={v} />
              ))}
            </div>
          )}
          <SimulationTranscriptView transcript={case_.transcript} />
        </CardContent>
      )}
    </Card>
  );
}
