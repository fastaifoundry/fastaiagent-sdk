import { Link } from "react-router-dom";
import { ExternalLink, Play } from "lucide-react";
import ReactDiffViewer, { DiffMethod } from "react-diff-viewer-continued";
import { CaseMediaPreview } from "@/components/evals/CaseMediaPreview";
import { VerdictTable } from "@/components/evals/VerdictTable";
import { caseValueText, comparableText, containsDescriptor } from "@/lib/evalDisplay";
import type { EvalCaseRow, EvalScorerDelta } from "@/lib/types";

interface Props {
  case_: EvalCaseRow;
  /** Optional: per-scorer deltas from compare view. */
  deltas?: EvalScorerDelta[];
  /**
   * Optional: the matched case from the other run, whose actual output is
   * shown next to this one. Its own run_id/case_id load its thumbnails.
   */
  otherCase?: EvalCaseRow;
  /** Optional: the other run's actual output, when no case row is at hand. */
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

const DIFF_STYLES = {
  variables: {
    dark: { diffViewerBackground: "transparent", gutterBackground: "transparent" },
    light: { diffViewerBackground: "transparent", gutterBackground: "transparent" },
  },
  contentText: { fontSize: 11 },
  gutter: { padding: "0 6px" },
};

/**
 * Two values side by side. Identical values get one "Identical" panel: a diff
 * of two equal strings is only a fold header ("@@ -0,1 +0,1 @@").
 */
function ValueDiff({
  left,
  right,
  leftLabel,
  rightLabel,
  dark,
}: {
  left: string;
  right: string;
  leftLabel: string;
  rightLabel: string;
  dark: boolean;
}) {
  const identical = left === right;
  if (identical) {
    return (
      <div className="rounded border" data-testid="identical-panel">
        <div className="border-b bg-muted/40 px-2 py-1 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
          {leftLabel} and {rightLabel} · Identical
        </div>
        <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words px-2 py-1.5 font-mono text-[11px]">
          {left || "(empty)"}
        </pre>
      </div>
    );
  }
  return (
    <div className="rounded border">
      <div className="grid grid-cols-2 border-b bg-muted/40 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        <div className="border-r px-2 py-1">{leftLabel}</div>
        <div className="px-2 py-1">{rightLabel}</div>
      </div>
      <div className="overflow-x-auto">
        <ReactDiffViewer
          oldValue={left}
          newValue={right}
          splitView
          compareMethod={DiffMethod.WORDS}
          hideLineNumbers
          showDiffOnly={!identical}
          useDarkTheme={dark}
          styles={DIFF_STYLES}
        />
      </div>
    </div>
  );
}

/**
 * Two values that hold images, PDFs or bytes, as thumbnails side by side.
 * Equality compares descriptors by sha256 and size, never their raw JSON.
 * Each side loads thumbnails from its own case.
 */
function MediaCompare({
  left,
  leftCase,
  leftLabel,
  right,
  rightCase,
  rightLabel,
}: {
  left: unknown;
  leftCase: EvalCaseRow;
  leftLabel: string;
  right: unknown;
  /** Absent when only the other value is known: it then shows as text. */
  rightCase?: EvalCaseRow;
  rightLabel: string;
}) {
  const identical = comparableText(left) === comparableText(right);
  return (
    <div className="rounded border" data-testid="media-compare">
      <div className="border-b bg-muted/40 px-2 py-1 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        {leftLabel} and {rightLabel} · {identical ? "Identical" : "Differ"}
      </div>
      <div className="grid grid-cols-1 gap-3 p-2 md:grid-cols-2">
        <div>
          <div className="mb-1 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
            {leftLabel}
          </div>
          <CaseMediaPreview value={left} runId={leftCase.run_id} caseId={leftCase.case_id} />
        </div>
        <div>
          <div className="mb-1 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
            {rightLabel}
          </div>
          {rightCase ? (
            <CaseMediaPreview value={right} runId={rightCase.run_id} caseId={rightCase.case_id} />
          ) : (
            <span className="font-mono">{caseValueText(right, 400)}</span>
          )}
        </div>
      </div>
    </div>
  );
}

export function CaseDiffCard({
  case_,
  deltas,
  otherCase,
  otherActual: otherActualProp,
  thisLabel = "This run",
  otherLabel = "Other run",
}: Props) {
  const otherActual = otherCase ? otherCase.actual_output : otherActualProp;
  const hasOther = otherCase !== undefined || otherActualProp !== undefined;
  const expected = stringifyForDiff(case_.expected_output);
  const actual = stringifyForDiff(case_.actual_output);
  const other = stringifyForDiff(otherActual);
  const dark =
    typeof document !== "undefined" &&
    document.documentElement.classList.contains("dark");
  // Images, PDFs and bytes are stored as descriptors: show their thumbnails,
  // not a diff of descriptor JSON.
  const hasMedia =
    containsDescriptor(case_.expected_output) || containsDescriptor(case_.actual_output);
  const otherHasMedia = containsDescriptor(case_.actual_output) || containsDescriptor(otherActual);
  const actualLabel = hasOther ? `Actual (${thisLabel})` : "Actual";

  return (
    <div className="rounded-md border bg-background p-3 space-y-3 text-xs">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            Case #{case_.ordinal}
          </div>
          {containsDescriptor(case_.input) ? (
            <div className="mt-1">
              <CaseMediaPreview value={case_.input} runId={case_.run_id} caseId={case_.case_id} />
            </div>
          ) : (
            <div className="mt-1 truncate font-mono">{caseValueText(case_.input, 200)}</div>
          )}
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

      {case_.error && (
        <div className="rounded border bg-muted px-2 py-1.5 font-mono text-muted-foreground">
          <span className="mr-2 uppercase">errored</span>
          {case_.error}
        </div>
      )}

      <VerdictTable
        perScorer={case_.per_scorer ?? {}}
        scoringTraceId={case_.scoring_trace_id}
        deltas={deltas}
      />

      {hasMedia ? (
        <MediaCompare
          left={case_.expected_output}
          leftCase={case_}
          leftLabel="Expected"
          right={case_.actual_output}
          rightCase={case_}
          rightLabel={actualLabel}
        />
      ) : (
        <ValueDiff
          left={expected}
          right={actual}
          leftLabel="Expected"
          rightLabel={actualLabel}
          dark={dark}
        />
      )}

      {hasOther &&
        (otherHasMedia ? (
          // Without the other case row there is no run/case id to load its
          // thumbnails from, so its side falls back to the one-line text.
          <MediaCompare
            left={case_.actual_output}
            leftCase={case_}
            leftLabel={`Actual (${thisLabel})`}
            right={otherActual}
            rightCase={otherCase}
            rightLabel={`Actual (${otherLabel})`}
          />
        ) : (
          <ValueDiff
            left={actual}
            right={other}
            leftLabel={`Actual (${thisLabel})`}
            rightLabel={`Actual (${otherLabel})`}
            dark={dark}
          />
        ))}
    </div>
  );
}
