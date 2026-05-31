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
import { formatTimeAgo } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { CriterionVerdict, SimulationCase } from "@/lib/types";

export function SimulationDetailPage() {
  const { runId } = useParams<{ runId: string }>();
  const [expanded, setExpanded] = useState<string | null>(null);
  const query = useSimulationRun(runId);

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

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatCard
          label="Pass rate"
          value={`${Math.round((run.pass_rate ?? 0) * 100)}%`}
        />
        <StatCard label="Passed" value={String(run.pass_count ?? 0)} />
        <StatCard label="Failed" value={String(run.fail_count ?? 0)} />
        <StatCard label="Started" value={formatTimeAgo(run.started_at ?? "")} />
      </div>

      {cases.length === 0 ? (
        <EmptyState title="No scenarios" description="This run has no cases." />
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
        verdict.passed
          ? "bg-fa-success/10 text-fa-success"
          : "bg-destructive/10 text-destructive"
      )}
      title={verdict.reason ?? undefined}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      <span className="font-mono opacity-70">{verdict.kind}</span>
      {verdict.criterion}
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
  const passed = case_.passed === 1;
  return (
    <Card>
      <CardHeader
        className="cursor-pointer flex-row items-center justify-between gap-3 py-3"
        onClick={onToggle}
      >
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
              passed
                ? "bg-fa-success/10 text-fa-success"
                : "bg-destructive/10 text-destructive"
            )}
          >
            {passed ? "PASS" : "FAIL"}
          </span>
        </div>
        <span className="text-xs text-muted-foreground">
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
