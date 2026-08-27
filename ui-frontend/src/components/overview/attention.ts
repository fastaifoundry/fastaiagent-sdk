/**
 * Derives the "needs attention" items for the Home page.
 *
 * Home used to open with six count tiles: all true, none of them actionable —
 * nothing on the landing page ever said something was *wrong*. This turns the
 * same numbers into statements, ranks them by severity, and points at the page
 * that owns the fix.
 *
 * Every signal here comes from the `/api/overview` payload the page already
 * fetches. No new requests, no new endpoints.
 *
 * Kept as a pure function (no JSX, no hooks) so the ranking rules can be
 * unit-tested directly — see attention.test.ts.
 */

export type Severity = "critical" | "warning" | "info";

export interface AttentionItem {
  key: string;
  severity: Severity;
  title: string;
  detail: string;
  cta: string;
  to: string;
}

/** The subset of the overview payload this module reads. */
export interface AttentionInput {
  failing_traces_last_24h: number;
  eval_runs_last_7d: number;
  avg_pass_rate_last_7d: number;
  pending_approvals_count: number;
  failed_executions_count: number;
  agents_with_errors: { agent_name: string; error_count: number }[];
}

/** Below this, a suite is not "mostly passing" any more. */
const PASS_RATE_FLOOR = 0.7;

const SEVERITY_ORDER: Record<Severity, number> = {
  critical: 0,
  warning: 1,
  info: 2,
};

const plural = (n: number, one: string, many: string) => (n === 1 ? one : many);

export function buildAttention(data: AttentionInput | undefined): AttentionItem[] {
  if (!data) return [];
  const items: AttentionItem[] = [];

  // A durable execution that failed or was interrupted is the most serious
  // thing this UI can report: work was accepted and then lost.
  if (data.failed_executions_count > 0) {
    const n = data.failed_executions_count;
    items.push({
      key: "failed-executions",
      severity: "critical",
      title: `${n} ${plural(n, "execution", "executions")} failed or interrupted`,
      detail:
        "Durable runs that stopped before completing. Resume them, or inspect the last checkpoint to see where they stopped.",
      cta: "Review executions",
      to: "/approvals",
    });
  }

  if (data.failing_traces_last_24h > 0) {
    const n = data.failing_traces_last_24h;
    // Name the worst offenders inline rather than emitting a card per agent —
    // they are all symptoms of the same failures.
    const top = [...data.agents_with_errors]
      .sort((a, b) => b.error_count - a.error_count)
      .slice(0, 3)
      .map((a) => a.agent_name);
    items.push({
      key: "failing-traces",
      severity: "warning",
      title: `${n} ${plural(n, "trace", "traces")} failed in the last 24h`,
      detail: top.length
        ? `Most recent errors involve ${top.join(", ")}.`
        : "Open the trace list filtered to errors to see what broke.",
      cta: "View failing traces",
      to: "/traces",
    });
  }

  // A blocked run is not an error, but it is stalled until a human acts.
  if (data.pending_approvals_count > 0) {
    const n = data.pending_approvals_count;
    items.push({
      key: "pending-approvals",
      severity: "warning",
      title: `${n} ${plural(n, "run is", "runs are")} waiting for approval`,
      detail:
        "These runs are paused at a human-in-the-loop checkpoint and will not continue until reviewed.",
      cta: "Review approvals",
      to: "/approvals",
    });
  }

  // Only meaningful if something actually ran — a 0% pass rate over zero runs
  // is not a regression, it is an empty dataset.
  if (data.eval_runs_last_7d > 0 && data.avg_pass_rate_last_7d < PASS_RATE_FLOOR) {
    const pct = Math.round(data.avg_pass_rate_last_7d * 100);
    items.push({
      key: "low-pass-rate",
      severity: "warning",
      title: `Eval pass rate is ${pct}%`,
      detail: `Averaged across ${data.eval_runs_last_7d} ${plural(
        data.eval_runs_last_7d,
        "run",
        "runs"
      )} in the last 7 days, below the ${Math.round(PASS_RATE_FLOOR * 100)}% floor.`,
      cta: "View eval runs",
      to: "/evals",
    });
  }

  return items.sort(
    (a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity]
  );
}
