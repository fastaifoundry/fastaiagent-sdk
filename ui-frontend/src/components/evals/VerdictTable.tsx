import { useState } from "react";
import { Link } from "react-router-dom";
import { ExternalLink } from "lucide-react";
import { cn } from "@/lib/utils";
import { verdictScoreText } from "@/lib/evalDisplay";
import {
  evalVerdictOutcome,
  type EvalScorerDelta,
  type EvalVerdict,
  type EvalVerdictOutcome,
} from "@/lib/types";

/** Reasons longer than this start collapsed. */
const REASON_FOLD = 300;

const OUTCOME_CHIP: Record<EvalVerdictOutcome, { label: string; className: string; title: string }> = {
  passed: {
    label: "passed",
    className: "bg-fa-success/10 text-fa-success",
    title: "The check ran and passed.",
  },
  failed: {
    label: "failed",
    className: "bg-destructive/10 text-destructive",
    title: "The check ran and failed.",
  },
  // Neutral grey, never red: the check could not run, so it says nothing
  // about the agent.
  errored: {
    label: "could not run",
    className: "bg-muted text-muted-foreground",
    title: "Errored: the check could not run (judge outage, unparseable verdict, or the scorer raised).",
  },
  skipped: {
    label: "n/a",
    className: "bg-transparent text-muted-foreground/70 border border-dashed",
    title: "n/a: the check does not apply to this case.",
  },
};

export function VerdictBadge({ outcome }: { outcome: EvalVerdictOutcome }) {
  const meta = OUTCOME_CHIP[outcome];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 font-mono text-[11px]",
        meta.className
      )}
      title={meta.title}
      data-outcome={outcome}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {meta.label}
    </span>
  );
}

function toolList(v: unknown): string | null {
  if (Array.isArray(v)) {
    return v.length ? v.map((x) => (typeof x === "string" ? x : JSON.stringify(x))).join(" → ") : "(none)";
  }
  if (typeof v === "number") return `${v} step${v === 1 ? "" : "s"}`;
  return null;
}

function Reason({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  if (text.length <= REASON_FOLD) {
    return <span className="whitespace-pre-wrap break-words">{text}</span>;
  }
  return (
    <span className="whitespace-pre-wrap break-words">
      {open ? text : `${text.slice(0, REASON_FOLD)}…`}{" "}
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="text-primary hover:underline"
      >
        {open ? "Show less" : "Show more"}
      </button>
    </span>
  );
}

interface Props {
  perScorer: Record<string, EvalVerdict>;
  /** The trace of this case's scoring (one span per scorer, judge calls beneath). */
  scoringTraceId?: string | null;
  /** From a compare view: rings the scorers whose verdict changed. */
  deltas?: EvalScorerDelta[];
}

/** One row per scorer: outcome, score, reason, and the scoring trace. */
export function VerdictTable({ perScorer, scoringTraceId, deltas }: Props) {
  const entries = Object.entries(perScorer ?? {});
  if (entries.length === 0) return null;
  return (
    <div className="overflow-x-auto rounded border">
      <table className="w-full text-xs" data-testid="verdict-table">
        <thead className="bg-muted/40 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
          <tr>
            <th className="px-2 py-1 text-left font-normal">Scorer</th>
            <th className="px-2 py-1 text-left font-normal">Outcome</th>
            <th className="px-2 py-1 text-left font-normal">Score</th>
            <th className="px-2 py-1 text-left font-normal">Reason</th>
            <th className="px-2 py-1 text-left font-normal">Trace</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([name, v]) => {
            const outcome = evalVerdictOutcome(v);
            const score = verdictScoreText(v);
            const changed = deltas?.find((d) => d.scorer === name)?.changed;
            const expected = toolList(v.details?.expected);
            const actual = toolList(v.details?.actual);
            return (
              <tr
                key={name}
                className={cn("border-t align-top", changed && "bg-primary/5")}
                data-testid={`verdict-row-${name}`}
              >
                <td className="px-2 py-1.5 font-mono">
                  {name}
                  {changed && (
                    <span className="ml-1 text-[10px] text-primary" title="Changed between the two runs">
                      changed
                    </span>
                  )}
                </td>
                <td className="px-2 py-1.5">
                  <VerdictBadge outcome={outcome} />
                </td>
                <td className="px-2 py-1.5 font-mono tabular-nums" data-testid={`verdict-score-${name}`}>
                  {score ?? <span className="text-muted-foreground">—</span>}
                </td>
                <td className="px-2 py-1.5 text-muted-foreground">
                  {v.reason ? <Reason text={v.reason} /> : "—"}
                  {(expected != null || actual != null) && (
                    <div className="mt-1 space-y-0.5 font-mono text-[11px]">
                      {expected != null && <div>expected: {expected}</div>}
                      {actual != null && <div>actual: {actual}</div>}
                    </div>
                  )}
                </td>
                <td className="px-2 py-1.5">
                  {scoringTraceId ? (
                    <Link
                      to={`/traces/${scoringTraceId}`}
                      title="The trace of this case's scoring"
                      className="inline-flex items-center gap-1 text-muted-foreground hover:text-primary"
                    >
                      scoring
                      <ExternalLink className="h-3 w-3" />
                    </Link>
                  ) : (
                    <span className="text-muted-foreground">—</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
