import { Link } from "react-router-dom";
import { ExternalLink, MessagesSquare, Scale, Shield, TrendingUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useTraceScores } from "@/hooks/use-analytics";
import { effectSuffix, outcomeMeta, TONE_CHIP } from "@/lib/guardrail";
import { cn } from "@/lib/utils";
import { evalVerdictOutcome, type EvalVerdict, type EvalVerdictOutcome } from "@/lib/types";

interface Props {
  traceId: string;
}

const VERDICT_CHIP: Record<EvalVerdictOutcome, { label: string; className: string }> = {
  passed: { label: "pass", className: "bg-fa-success/10 text-fa-success" },
  failed: { label: "fail", className: "bg-destructive/10 text-destructive" },
  // Neutral: the check could not run, which says nothing about the agent.
  errored: { label: "could not run", className: "bg-muted text-muted-foreground" },
  skipped: { label: "n/a", className: "bg-muted/50 text-muted-foreground/70" },
};

/** One chip per scorer; errored / n/a show no score (theirs is a placeholder). */
function VerdictChips({ perScorer }: { perScorer: Record<string, EvalVerdict> | null | undefined }) {
  return (
    <>
      {Object.entries(perScorer ?? {}).map(([name, v]) => {
        const outcome = evalVerdictOutcome(v);
        const chip = VERDICT_CHIP[outcome];
        const scored = outcome === "passed" || outcome === "failed";
        return (
          <span
            key={name}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[11px] font-mono",
              chip.className
            )}
            title={v?.reason ?? undefined}
            data-outcome={outcome}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-current" />
            {name} {chip.label}
            {scored && typeof v.score === "number" && (
              <span className="opacity-70 tabular-nums">{v.score.toFixed(2)}</span>
            )}
          </span>
        );
      })}
    </>
  );
}

/**
 * Scores surface for a single trace — aggregates guardrail events and eval
 * cases that point at this trace so users see all "was this good?" signals
 * in one place. Shown beneath the span tree on Trace Detail.
 */
export function TraceScoresCard({ traceId }: Props) {
  const scores = useTraceScores(traceId);
  const guardrails = scores.data?.guardrail_events ?? [];
  const evalCases = scores.data?.eval_cases ?? [];
  const evalScoring = scores.data?.eval_scoring ?? [];
  const simCases = scores.data?.simulation_cases ?? [];

  if (scores.isLoading) return null;
  if (
    guardrails.length === 0 &&
    evalCases.length === 0 &&
    evalScoring.length === 0 &&
    simCases.length === 0
  )
    return null;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm">Scores</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        {guardrails.length > 0 && (
          <div>
            <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
              <Shield className="h-3 w-3" />
              Guardrail events
            </div>
            <ul className="space-y-1.5">
              {guardrails.map((g) => {
                const meta = outcomeMeta(g.outcome);
                const suffix = effectSuffix(g.outcome, g.effect);
                return (
                  <li key={g.event_id}>
                    <Link
                      to={`/guardrail-events/${g.event_id}`}
                      className="flex items-center justify-between rounded-md border px-2 py-1.5 transition-colors hover:border-primary/40"
                    >
                      <div className="flex items-center gap-2">
                        <span
                          className={cn(
                            "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[11px] font-mono uppercase",
                            TONE_CHIP[meta.tone]
                          )}
                          data-tone={meta.tone}
                        >
                          <span className="h-1.5 w-1.5 rounded-full bg-current" />
                          {meta.label}
                        </span>
                        {suffix && (
                          <span className="font-mono text-[11px] text-muted-foreground">
                            · {suffix}
                          </span>
                        )}
                        <span className="font-mono text-xs">{g.guardrail_name}</span>
                        {g.message && (
                          <span className="truncate text-xs text-muted-foreground">
                            {g.message}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs tabular-nums text-muted-foreground">
                          {g.score != null ? g.score.toFixed(2) : ""}
                        </span>
                        <ExternalLink className="h-3 w-3 text-muted-foreground" />
                      </div>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        )}

        {evalCases.length > 0 && (
          <div>
            <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
              <TrendingUp className="h-3 w-3" />
              Eval cases using this trace
            </div>
            <ul className="space-y-1.5">
              {evalCases.map((c) => (
                <li
                  key={c.case_id}
                  className="flex items-center justify-between rounded-md border px-2 py-1.5"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <VerdictChips perScorer={c.per_scorer} />
                    <span className="text-xs text-muted-foreground">
                      {c.run_name ? `in ${c.run_name}` : `run ${c.run_id.slice(0, 8)}`}
                    </span>
                  </div>
                  <Link
                    to={`/evals/${c.run_id}`}
                    className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-primary"
                  >
                    open run
                    <ExternalLink className="h-3 w-3" />
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        )}

        {evalScoring.length > 0 && (
          <div>
            <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
              <Scale className="h-3 w-3" />
              This trace is an eval scoring
            </div>
            <ul className="space-y-1.5">
              {evalScoring.map((c) => (
                <li
                  key={c.case_id}
                  className="flex items-center justify-between gap-2 rounded-md border px-2 py-1.5"
                >
                  <div className="flex min-w-0 flex-wrap items-center gap-2">
                    <Link
                      to={`/evals/${c.run_id}`}
                      className="text-xs hover:text-primary hover:underline"
                    >
                      Scoring for case #{c.ordinal} of run{" "}
                      {c.run_name || c.run_id.slice(0, 8)}
                    </Link>
                    <VerdictChips perScorer={c.per_scorer} />
                  </div>
                  {c.trace_id && (
                    <Link
                      to={`/traces/${c.trace_id}`}
                      className="inline-flex shrink-0 items-center gap-1 text-xs text-muted-foreground hover:text-primary"
                      title="The agent trace this scoring judged"
                    >
                      agent trace
                      <ExternalLink className="h-3 w-3" />
                    </Link>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}

        {simCases.length > 0 && (
          <div>
            <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
              <MessagesSquare className="h-3 w-3" />
              Simulated conversation
            </div>
            <ul className="space-y-1.5">
              {simCases.map((c) => {
                const errored = !!c.error;
                const passed = !errored && (c.passed === 1 || c.passed === true);
                return (
                  <li
                    key={c.case_id}
                    className="flex items-center justify-between gap-2 rounded-md border px-2 py-1.5"
                  >
                    <div className="flex min-w-0 flex-wrap items-center gap-2">
                      <span
                        className={cn(
                          "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[11px] font-mono",
                          errored
                            ? "bg-muted text-muted-foreground"
                            : passed
                            ? "bg-fa-success/10 text-fa-success"
                            : "bg-destructive/10 text-destructive"
                        )}
                        title={c.error ?? undefined}
                        data-testid={`sim-outcome-${c.case_id}`}
                      >
                        <span className="h-1.5 w-1.5 rounded-full bg-current" />
                        {errored ? "could not run" : passed ? "passed" : "failed"}
                      </span>
                      <Link
                        to={`/simulations/${c.run_id}`}
                        className="text-xs hover:text-primary hover:underline"
                      >
                        Simulated conversation: scenario {c.scenario_name ?? `#${c.ordinal}`} of run{" "}
                        {c.run_name || c.run_id.slice(0, 8)}
                      </Link>
                    </div>
                    <ExternalLink className="h-3 w-3 shrink-0 text-muted-foreground" />
                  </li>
                );
              })}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
