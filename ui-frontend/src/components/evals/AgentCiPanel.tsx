import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { Copy, GitBranch } from "lucide-react";
import { toast } from "sonner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { copyToClipboard } from "@/lib/format";
import { formatRate, formatRateDelta } from "@/lib/evalDisplay";
import { cn } from "@/lib/utils";
import type { EvalGate } from "@/lib/types";

const GATE_CHIP: Record<string, string> = {
  passed: "bg-fa-success/10 text-fa-success",
  failed: "bg-destructive/10 text-destructive",
  invalid: "bg-fa-warning/10 text-fa-warning",
};

/** The gate's verdict as a chip; unknown values render neutral. */
export function GateBadge({ outcome }: { outcome: string | null | undefined }) {
  if (!outcome) return <span className="text-muted-foreground">—</span>;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 font-mono text-[11px] uppercase",
        GATE_CHIP[outcome] ?? "bg-muted text-muted-foreground"
      )}
      title={
        outcome === "invalid"
          ? "Invalid: too many cases errored, or none was scored, for the verdict to count."
          : `Gate ${outcome}`
      }
      data-gate={outcome}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {outcome}
    </span>
  );
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function Label({ children }: { children: ReactNode }) {
  return (
    <div className="mb-1 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
      {children}
    </div>
  );
}

/**
 * What the Agent-CI gate decided about this run and why: the thresholds with
 * the values reached, the errored-check bound, why a run was invalid, the
 * baseline comparison and the commit. Every part is optional.
 */
export function AgentCiPanel({ gate, runId }: { gate: EvalGate; runId: string }) {
  const checks = Array.isArray(gate.checks) ? gate.checks : [];
  const thresholds =
    gate.thresholds && typeof gate.thresholds === "object" ? gate.thresholds : {};
  const rows =
    checks.length > 0
      ? checks.map((c) => ({
          key: `${c.metric}.${c.field}`,
          minimum: num(c.minimum),
          actual: num(c.actual),
          passed: c.passed as boolean | null,
        }))
      : Object.entries(thresholds).map(([key, min]) => ({
          key,
          minimum: num(min),
          actual: null,
          passed: null,
        }));
  const err = gate.error && typeof gate.error === "object" ? gate.error : null;
  const reasons = Array.isArray(gate.invalid_reasons) ? gate.invalid_reasons : [];
  const baseline = gate.baseline && typeof gate.baseline === "object" ? gate.baseline : null;
  const sha = typeof gate.git_sha === "string" && gate.git_sha ? gate.git_sha : null;
  const branch = typeof gate.git_branch === "string" && gate.git_branch ? gate.git_branch : null;

  return (
    <Card data-testid="agent-ci-panel">
      <CardHeader className="pb-3">
        <CardTitle className="flex flex-wrap items-center gap-2 text-sm">
          Agent CI gate
          <GateBadge outcome={gate.outcome} />
          {(sha || branch) && (
            <span className="ml-auto inline-flex items-center gap-1.5 text-xs font-normal text-muted-foreground">
              <GitBranch className="h-3.5 w-3.5" />
              {branch && <span className="font-mono">{branch}</span>}
              {sha && (
                <>
                  <span className="font-mono" title={sha}>
                    {sha.slice(0, 7)}
                  </span>
                  <button
                    type="button"
                    aria-label="Copy commit SHA"
                    title="Copy commit SHA"
                    className="inline-flex h-5 w-5 items-center justify-center rounded hover:bg-muted"
                    onClick={async () => {
                      try {
                        await copyToClipboard(sha);
                        toast.success("Commit SHA copied");
                      } catch {
                        toast.error("Could not copy");
                      }
                    }}
                  >
                    <Copy className="h-3 w-3" />
                  </button>
                </>
              )}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-1 gap-4 text-xs md:grid-cols-2">
        <div>
          <Label>Thresholds</Label>
          {rows.length === 0 ? (
            <p className="text-muted-foreground">No threshold was demanded.</p>
          ) : (
            <table className="w-full" data-testid="gate-thresholds">
              <thead className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
                <tr>
                  <th className="py-1 text-left font-normal">Metric</th>
                  <th className="py-1 text-right font-normal">Required</th>
                  <th className="py-1 text-right font-normal">Actual</th>
                  <th className="py-1 text-right font-normal" />
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.key} className="border-t">
                    <td className="py-1 font-mono">{r.key}</td>
                    <td className="py-1 text-right font-mono tabular-nums">
                      {r.minimum != null ? `≥ ${r.minimum}` : "—"}
                    </td>
                    <td className="py-1 text-right font-mono tabular-nums">
                      {r.actual != null ? r.actual.toFixed(4) : "—"}
                    </td>
                    <td className="py-1 pl-2 text-right">
                      {r.passed === true ? (
                        <span className="text-fa-success">pass</span>
                      ) : r.passed === false ? (
                        <span
                          className="text-destructive"
                          title={
                            r.actual == null
                              ? "No value to compare: the metric is not in this run, or none of its verdicts applied or could run."
                              : undefined
                          }
                        >
                          {r.actual == null ? "no value" : "fail"}
                        </span>
                      ) : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div className="space-y-4">
          {err && (
            <div>
              <Label>Errored cases</Label>
              <p className="font-mono" data-testid="gate-error-bound">
                {num(err.errored) ?? 0} errored
                {num(err.scored) != null && ` of ${(num(err.errored) ?? 0) + num(err.scored)!} cases`}
                {num(err.error_rate) != null && ` (${formatRate(num(err.error_rate))})`}
                {" · "}
                {num(err.max_error_rate) != null
                  ? `allowed ≤ ${formatRate(num(err.max_error_rate))}`
                  : "no bound set"}
              </p>
            </div>
          )}

          {reasons.length > 0 && (
            <div>
              <Label>Why the run is invalid</Label>
              <ul className="list-disc space-y-0.5 pl-4">
                {reasons.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          )}

          {baseline && (
            <div data-testid="gate-baseline">
              <Label>Baseline</Label>
              <div className="space-y-1">
                {baseline.run_id ? (
                  <div className="flex flex-wrap items-center gap-2">
                    <Link
                      to={`/evals/${baseline.run_id}`}
                      className="font-mono hover:text-primary hover:underline"
                    >
                      {baseline.run_name || baseline.run_id.slice(0, 12)}
                    </Link>
                    <Link
                      to={`/evals/compare?a=${encodeURIComponent(
                        baseline.run_id
                      )}&b=${encodeURIComponent(runId)}`}
                      className="text-primary hover:underline"
                    >
                      Compare
                    </Link>
                  </div>
                ) : (
                  <span className="text-muted-foreground">Baseline run not recorded</span>
                )}
                <p className="font-mono text-muted-foreground">
                  {[
                    baseline.case_pass_rate_delta != null
                      ? `cases Δ ${formatRateDelta(baseline.case_pass_rate_delta)}`
                      : null,
                    baseline.pass_rate_delta != null
                      ? `checks Δ ${formatRateDelta(baseline.pass_rate_delta)}`
                      : null,
                    baseline.regressed_count != null
                      ? `${baseline.regressed_count} regressed`
                      : null,
                    baseline.errored_count ? `${baseline.errored_count} errored` : null,
                    baseline.unmatched_count ? `${baseline.unmatched_count} unmatched` : null,
                    baseline.tolerance != null ? `tolerance ${baseline.tolerance}` : null,
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                </p>
                {baseline.pass_rate_comparable === false && (
                  <p className="text-fa-warning">
                    The baseline was stored under pre-1.89 rules, so the checks Δ is not like for
                    like.
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
