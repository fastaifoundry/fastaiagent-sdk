import { Link } from "react-router-dom";
import { ExternalLink, Play } from "lucide-react";
import ReactDiffViewer, { DiffMethod } from "react-diff-viewer-continued";
import { cn } from "@/lib/utils";
import type { EvalCaseRow, EvalScorerDelta } from "@/lib/types";

interface Props {
  case_: EvalCaseRow;
  /** Optional: per-scorer deltas from compare view. */
  deltas?: EvalScorerDelta[];
  /** Optional: show the "other run"'s actual output next to this one. */
  otherActual?: unknown;
  /** Label for the "this run" column when ``otherActual`` is set. */
  thisLabel?: string;
  /** Label for the "other run" column. */
  otherLabel?: string;
}

function stringifyForDiff(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function CaseDiffCard({
  case_,
  deltas,
  otherActual,
  thisLabel = "This run",
  otherLabel = "Other run",
}: Props) {
  const expected = stringifyForDiff(case_.expected_output);
  const actual = stringifyForDiff(case_.actual_output);
  const other = stringifyForDiff(otherActual);
  const dark =
    typeof document !== "undefined" &&
    document.documentElement.classList.contains("dark");

  const perScorer = case_.per_scorer ?? {};

  return (
    <div className="rounded-md border bg-background p-3 space-y-3 text-xs">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            Case #{case_.ordinal}
          </div>
          <div className="mt-1 truncate font-mono">
            {stringifyForDiff(case_.input).slice(0, 200)}
          </div>
        </div>
        {case_.trace_id && (
          <div className="flex shrink-0 items-center gap-2">
            <Link
              to={`/traces/${case_.trace_id}`}
              title="Open trace"
              className="inline-flex items-center gap-1 rounded border px-2 py-0.5 text-[11px] text-muted-foreground hover:border-primary hover:text-primary"
            >
              <ExternalLink className="h-3 w-3" />
              Trace
            </Link>
            <Link
              to={`/traces/${case_.trace_id}/replay`}
              title="Open in Replay"
              className="inline-flex items-center gap-1 rounded border px-2 py-0.5 text-[11px] text-muted-foreground hover:border-primary hover:text-primary"
            >
              <Play className="h-3 w-3" />
              Replay
            </Link>
          </div>
        )}
      </div>

      {Object.keys(perScorer).length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(perScorer).map(([scorer, r]) => {
            const delta = deltas?.find((d) => d.scorer === scorer);
            const passed = r?.passed;
            return (
              <span
                key={scorer}
                className={cn(
                  "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 font-mono text-[10px]",
                  passed
                    ? "bg-fa-success/10 text-fa-success"
                    : "bg-destructive/10 text-destructive",
                  delta?.changed && "ring-1 ring-primary"
                )}
                title={r?.reason ?? undefined}
              >
                {scorer} · {passed ? "pass" : "fail"}
                {typeof r?.score === "number" && (
                  <span className="opacity-70">{r.score.toFixed(2)}</span>
                )}
              </span>
            );
          })}
        </div>
      )}

      <div className="rounded border">
        <div className="grid grid-cols-2 border-b bg-muted/40 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
          <div className="border-r px-2 py-1">Expected</div>
          <div className="px-2 py-1">
            {otherActual !== undefined ? `Actual (${thisLabel})` : "Actual"}
          </div>
        </div>
        <div className="overflow-x-auto">
          <ReactDiffViewer
            oldValue={expected}
            newValue={actual}
            splitView
            compareMethod={DiffMethod.WORDS}
            hideLineNumbers
            useDarkTheme={dark}
            styles={{
              variables: {
                dark: {
                  diffViewerBackground: "transparent",
                  gutterBackground: "transparent",
                },
                light: {
                  diffViewerBackground: "transparent",
                  gutterBackground: "transparent",
                },
              },
              contentText: { fontSize: 11 },
              gutter: { padding: "0 6px" },
            }}
          />
        </div>
      </div>

      {otherActual !== undefined && (
        <div className="rounded border">
          <div className="grid grid-cols-2 border-b bg-muted/40 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
            <div className="border-r px-2 py-1">
              Actual ({thisLabel})
            </div>
            <div className="px-2 py-1">Actual ({otherLabel})</div>
          </div>
          <div className="overflow-x-auto">
            <ReactDiffViewer
              oldValue={actual}
              newValue={other}
              splitView
              compareMethod={DiffMethod.WORDS}
              hideLineNumbers
              useDarkTheme={dark}
              styles={{
                variables: {
                  dark: {
                    diffViewerBackground: "transparent",
                    gutterBackground: "transparent",
                  },
                  light: {
                    diffViewerBackground: "transparent",
                    gutterBackground: "transparent",
                  },
                },
                contentText: { fontSize: 11 },
                gutter: { padding: "0 6px" },
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
