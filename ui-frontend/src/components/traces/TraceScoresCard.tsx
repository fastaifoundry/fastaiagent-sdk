import { Link } from "react-router-dom";
import { ExternalLink, Shield, TrendingUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useTraceScores } from "@/hooks/use-analytics";
import { cn } from "@/lib/utils";

interface Props {
  traceId: string;
}

const OUTCOME_CLASS: Record<string, string> = {
  passed: "bg-fa-success/10 text-fa-success",
  blocked: "bg-destructive/10 text-destructive",
  warned: "bg-fa-warning/10 text-fa-warning",
};

/**
 * Scores surface for a single trace — aggregates guardrail events and eval
 * cases that point at this trace so users see all "was this good?" signals
 * in one place. Shown beneath the span tree on Trace Detail.
 */
export function TraceScoresCard({ traceId }: Props) {
  const scores = useTraceScores(traceId);
  const guardrails = scores.data?.guardrail_events ?? [];
  const evalCases = scores.data?.eval_cases ?? [];

  if (scores.isLoading) return null;
  if (guardrails.length === 0 && evalCases.length === 0) return null;

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
                const outcome = (g.outcome ?? "").toLowerCase();
                const cls = OUTCOME_CLASS[outcome] ?? "bg-muted text-muted-foreground";
                return (
                  <li
                    key={g.event_id}
                    className="flex items-center justify-between rounded-md border px-2 py-1.5"
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className={cn(
                          "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[11px] font-mono uppercase",
                          cls
                        )}
                      >
                        <span className="h-1.5 w-1.5 rounded-full bg-current" />
                        {outcome || "unknown"}
                      </span>
                      <span className="font-mono text-xs">{g.guardrail_name}</span>
                      {g.message && (
                        <span className="truncate text-xs text-muted-foreground">
                          {g.message}
                        </span>
                      )}
                    </div>
                    <span className="font-mono text-xs tabular-nums text-muted-foreground">
                      {g.score != null ? g.score.toFixed(2) : ""}
                    </span>
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
              {evalCases.map((c) => {
                const scorerEntries = Object.entries(c.per_scorer ?? {});
                return (
                  <li
                    key={c.case_id}
                    className="flex items-center justify-between rounded-md border px-2 py-1.5"
                  >
                    <div className="flex items-center gap-2">
                      {scorerEntries.map(([name, score]) => (
                        <span
                          key={name}
                          className={cn(
                            "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[11px] font-mono",
                            score.passed
                              ? "bg-fa-success/10 text-fa-success"
                              : "bg-destructive/10 text-destructive"
                          )}
                        >
                          <span className="h-1.5 w-1.5 rounded-full bg-current" />
                          {name} {score.passed ? "pass" : "fail"}
                          <span className="opacity-70 tabular-nums">
                            {(score.score ?? 0).toFixed(2)}
                          </span>
                        </span>
                      ))}
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
                );
              })}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
