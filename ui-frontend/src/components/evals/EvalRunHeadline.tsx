import { StatCard } from "@/components/shared/StatCard";
import { caseCounts, costLabel, formatRate } from "@/lib/evalDisplay";
import { formatDateTime, formatDurationMs } from "@/lib/format";
import type { EvalRunRow } from "@/lib/types";

const CHECKS_TITLE =
  "Checks = passed verdicts / (passed + failed verdicts). This is overall.pass_rate, what --eval-fail-under gates. Checks that could not run (errored) or do not apply (n/a) are counted, never rated.";

function durationMs(run: Pick<EvalRunRow, "started_at" | "finished_at">): number | null {
  if (!run.started_at || !run.finished_at) return null;
  const ms = new Date(run.finished_at).getTime() - new Date(run.started_at).getTime();
  return Number.isFinite(ms) && ms >= 0 ? ms : null;
}

/**
 * A run's headline numbers, cases first (D1): cases passed, failed and
 * errored, then the checks rate the gate reads, then cost and timing.
 * Counts come from the run row, never the filtered case list.
 */
export function EvalRunHeadline({ run }: { run: EvalRunRow }) {
  const c = caseCounts(run);
  const checkPassed = run.pass_count ?? 0;
  const checkFailed = run.fail_count ?? 0;
  const checkHint = [
    `${checkPassed}/${checkPassed + checkFailed}`,
    run.check_skip_count ? `${run.check_skip_count} n/a` : null,
    run.check_error_count ? `${run.check_error_count} errored` : null,
  ]
    .filter(Boolean)
    .join(" · ");
  const agent = costLabel(run.agent_cost_usd, run.agent_cost_known);
  const evalCost = costLabel(run.eval_cost_usd, run.eval_cost_known);
  const took = durationMs(run);
  const timing = [
    took != null ? `took ${formatDurationMs(took)}` : null,
    run.avg_latency_ms ? `avg case ${formatDurationMs(run.avg_latency_ms)}` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-4 xl:grid-cols-7" data-testid="eval-headline">
      <StatCard
        label="Cases passed"
        value={formatRate(run.case_pass_rate)}
        hint={`${c.passed}/${c.judged}`}
        title="Passed cases / (passed + failed cases). Errored cases are left out."
        valueTitle={run.case_pass_rate == null ? "No case reached a verdict" : undefined}
      />
      <StatCard label="Failed" value={String(c.failed)} hint="cases" />
      <StatCard
        label="Errored"
        value={String(c.errored)}
        hint="could not run"
        title="Cases with no verdict: the agent failed, a check could not run, or no check applied."
      />
      <StatCard
        label="Checks"
        value={formatRate(run.pass_rate)}
        hint={checkHint}
        title={CHECKS_TITLE}
      />
      <StatCard
        label="Agent cost"
        value={agent.text}
        valueTitle={agent.title}
        hint={agent.partial ? "partial" : undefined}
      />
      <StatCard
        label="Eval cost"
        value={evalCost.text}
        valueTitle={evalCost.title}
        hint={evalCost.partial ? "partial" : undefined}
      />
      <StatCard label="Started" value={formatDateTime(run.started_at ?? "")} hint={timing} />
    </div>
  );
}
