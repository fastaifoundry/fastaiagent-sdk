import { useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import ReactDiffViewer, { DiffMethod } from "react-diff-viewer-continued";
import { JsonViewer } from "@/components/shared/JsonViewer";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { ComparisonResult, ReplayStep, ToolDiff } from "@/lib/types";

interface Props {
  comparison: ComparisonResult;
  originalOutput: unknown;
  newOutput: unknown;
}

function safeStep(
  steps: ReplayStep[],
  idx: number
): ReplayStep | null {
  return steps[idx] ?? null;
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

function contentsDiffer(a: unknown, b: unknown): boolean {
  return stringifyForDiff(a) !== stringifyForDiff(b);
}

/**
 * The rows to show, as ``[originalIndex, newIndex]`` pairs.
 *
 * Since 1.89.0 the server aligns only the run's own turns (memory and nested
 * spans are skipped) and returns the pairs; ``diverged_at`` indexes that
 * alignment's original side. An older server sends no pairs, so fall back to
 * pairing by position.
 */
function rowPairs(comparison: ComparisonResult): [number | null, number | null][] {
  if (comparison.aligned && comparison.aligned.length > 0) return comparison.aligned;
  const max = Math.max(comparison.original_steps.length, comparison.new_steps.length);
  return Array.from({ length: max }, (_, i) => [
    i < comparison.original_steps.length ? i : null,
    i < comparison.new_steps.length ? i : null,
  ]);
}

export function ReplayDiffView({
  comparison,
  originalOutput,
  newOutput,
}: Props) {
  const pairs = rowPairs(comparison);
  const diverged = comparison.diverged_at;
  // First row at or after the divergence; every later row is highlighted.
  // A row with no original step means the original ran fewer turns, so the
  // runs have diverged by then at the latest.
  const divergedRow =
    diverged == null ? -1 : pairs.findIndex(([o]) => o == null || o >= diverged);
  const toolDiffs = comparison.tool_diffs ?? [];
  const toolsDiverged = comparison.tools_diverged_at ?? null;
  const skipped = comparison.skipped_steps?.length ?? 0;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader className="flex flex-row items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
            <CardTitle className="text-sm">Original output</CardTitle>
          </CardHeader>
          <CardContent>
            <JsonViewer data={originalOutput ?? null} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-primary" />
            <CardTitle className="text-sm">Rerun output</CardTitle>
          </CardHeader>
          <CardContent>
            <JsonViewer data={newOutput ?? null} />
          </CardContent>
        </Card>
      </div>

      <div className="rounded-md border bg-card">
        <div className="flex items-center justify-between border-b px-4 py-2">
          <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
            Step-by-step comparison
            {skipped > 0 && (
              <span
                className="ml-2 normal-case tracking-normal"
                title="Memory spans and spans nested under a tool are not compared"
              >
                ({skipped} memory/nested step{skipped === 1 ? "" : "s"} not compared)
              </span>
            )}
          </span>
          <div className="flex items-center gap-1.5">
            {comparison.compare_status === "rerun_failed" ? (
              <span className="rounded-md bg-destructive/10 px-2 py-0.5 text-xs font-mono font-medium text-destructive">
                rerun trace failed to load
              </span>
            ) : diverged != null ? (
              <span className="rounded-md bg-primary/10 px-2 py-0.5 text-xs font-mono font-medium text-primary">
                diverged at step {diverged}
              </span>
            ) : null}
            {comparison.compare_status !== "rerun_failed" && toolsDiverged != null && (
              <span className="rounded-md bg-fa-warning/10 px-2 py-0.5 text-xs font-mono font-medium text-fa-warning">
                tools differ at step {toolsDiverged}
              </span>
            )}
          </div>
        </div>
        <div className="divide-y">
          {pairs.map(([o, n], row) => (
            <StepRow
              key={`${o ?? "-"}:${n ?? "-"}`}
              index={o ?? n ?? row}
              original={o == null ? null : safeStep(comparison.original_steps, o)}
              rerun={n == null ? null : safeStep(comparison.new_steps, n)}
              diverged={divergedRow >= 0 && row >= divergedRow}
              toolChanged={toolsDiverged != null && o === toolsDiverged}
            />
          ))}
        </div>
      </div>

      {toolDiffs.length > 0 && <ToolDiffList diffs={toolDiffs} />}
    </div>
  );
}

function resultPreview(value: unknown): string {
  const text = stringifyForDiff(value);
  return text.length > 300 ? `${text.slice(0, 300)}…` : text;
}

/**
 * Tool calls whose status or result changed (1.89.0). Kept apart from the
 * step diff: a recorded rerun with a blocked tool still reports the model's
 * answers as identical, and this is where the blocked call shows.
 */
function ToolDiffList({ diffs }: { diffs: ToolDiff[] }) {
  return (
    <div className="rounded-md border bg-card" data-testid="tool-diffs">
      <div className="border-b px-4 py-2 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        Tool calls that changed
      </div>
      <div className="divide-y">
        {diffs.map((d, i) => (
          <div
            key={`${d.name}-${d.original_step ?? "x"}-${d.new_step ?? "x"}-${i}`}
            className="space-y-2 px-4 py-3"
            data-testid={`tool-diff-${d.name}`}
          >
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="font-mono font-medium">{d.name}</span>
              <span className="font-mono text-muted-foreground">
                step {d.original_step ?? "—"} → {d.new_step ?? "—"}
              </span>
              {d.disposition && (
                <span className="rounded border px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground">
                  {d.disposition}
                </span>
              )}
            </div>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              <ToolDiffSide label="Original" status={d.original_status} result={d.original_result} />
              <ToolDiffSide label="Rerun" status={d.new_status} result={d.new_result} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ToolDiffSide({
  label,
  status,
  result,
}: {
  label: string;
  status: string | null;
  result: unknown;
}) {
  return (
    <div className="min-w-0 space-y-1">
      <span className="block text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        {label} · <span className="normal-case tracking-normal">{status ?? "no status"}</span>
      </span>
      <pre className="max-h-40 overflow-auto rounded-md border bg-muted/20 p-2 font-mono text-[11px] whitespace-pre-wrap">
        {resultPreview(result) || "—"}
      </pre>
    </div>
  );
}

function StepRow({
  index,
  original,
  rerun,
  diverged,
  toolChanged = false,
}: {
  index: number;
  original: ReplayStep | null;
  rerun: ReplayStep | null;
  diverged: boolean;
  toolChanged?: boolean;
}) {
  const nameA = original?.span_name ?? "—";
  const nameB = rerun?.span_name ?? "—";
  const nameChanged = nameA !== nameB;

  const inputChanged = contentsDiffer(original?.input, rerun?.input);
  const outputChanged = contentsDiffer(original?.output, rerun?.output);
  const hasAnyDiff = nameChanged || inputChanged || outputChanged;

  // Auto-expand the first diverged row so the user sees the actual diff
  // without hunting. Subsequent diverged rows stay collapsed to keep the
  // page readable.
  const [open, setOpen] = useState(false);
  const canExpand = hasAnyDiff && (original || rerun);

  return (
    <div
      className={cn(
        "grid grid-cols-[48px_1fr_1fr] items-start gap-3 border-l-2 px-4 py-3",
        diverged ? "border-primary bg-primary/5" : "border-transparent"
      )}
    >
      <div className="flex items-center gap-1 font-mono text-xs text-muted-foreground">
        {canExpand ? (
          <button
            type="button"
            aria-label={open ? "Collapse step diff" : "Expand step diff"}
            onClick={() => setOpen((v) => !v)}
            className="inline-flex h-4 w-4 items-center justify-center rounded hover:bg-muted"
          >
            {open ? (
              <ChevronDown className="h-3 w-3" />
            ) : (
              <ChevronRight className="h-3 w-3" />
            )}
          </button>
        ) : (
          <span className="inline-block h-4 w-4" />
        )}
        {index}
      </div>

      <StepColumn label="Original" step={original} highlight={nameChanged} />
      <StepColumn label="Rerun" step={rerun} highlight={nameChanged} />

      {toolChanged && (
        <span className="col-span-2 col-start-2 text-[11px] font-mono text-fa-warning">
          First tool call that changed: see "Tool calls that changed" below.
        </span>
      )}

      {open && canExpand && (
        <div className="col-span-3 mt-2 space-y-3">
          {inputChanged && (
            <DiffBlock
              label="Input"
              oldText={stringifyForDiff(original?.input)}
              newText={stringifyForDiff(rerun?.input)}
            />
          )}
          {outputChanged && (
            <DiffBlock
              label="Output"
              oldText={stringifyForDiff(original?.output)}
              newText={stringifyForDiff(rerun?.output)}
            />
          )}
          {!inputChanged && !outputChanged && (
            <p className="text-[11px] italic text-muted-foreground">
              Span names differ but inputs/outputs captured on this step are
              identical.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function StepColumn({
  label,
  step,
  highlight,
}: {
  label: string;
  step: ReplayStep | null;
  highlight: boolean;
}) {
  return (
    <div className="min-w-0">
      <span className="mb-0.5 block text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        {label}
      </span>
      <span
        className={cn(
          "block truncate font-mono text-xs",
          highlight ? "text-primary font-semibold" : "text-foreground"
        )}
      >
        {step?.span_name ?? "—"}
      </span>
    </div>
  );
}

function DiffBlock({
  label,
  oldText,
  newText,
}: {
  label: string;
  oldText: string;
  newText: string;
}) {
  return (
    <div className="rounded-md border bg-background">
      <div className="border-b px-3 py-1.5 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        {label}
      </div>
      <div className="overflow-x-auto text-xs">
        <ReactDiffViewer
          oldValue={oldText}
          newValue={newText}
          splitView
          compareMethod={DiffMethod.WORDS}
          hideLineNumbers={false}
          useDarkTheme={
            typeof document !== "undefined" &&
            document.documentElement.classList.contains("dark")
          }
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
  );
}
