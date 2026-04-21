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
import type { ComparisonResult, ReplayStep } from "@/lib/types";

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

export function ReplayDiffView({
  comparison,
  originalOutput,
  newOutput,
}: Props) {
  const maxSteps = Math.max(
    comparison.original_steps.length,
    comparison.new_steps.length
  );
  const diverged = comparison.diverged_at;

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
          </span>
          {diverged != null && (
            <span className="rounded-md bg-primary/10 px-2 py-0.5 text-xs font-mono font-medium text-primary">
              diverged at step {diverged}
            </span>
          )}
        </div>
        <div className="divide-y">
          {Array.from({ length: maxSteps }, (_, idx) => (
            <StepRow
              key={idx}
              index={idx}
              original={safeStep(comparison.original_steps, idx)}
              rerun={safeStep(comparison.new_steps, idx)}
              diverged={diverged != null && idx >= diverged}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function StepRow({
  index,
  original,
  rerun,
  diverged,
}: {
  index: number;
  original: ReplayStep | null;
  rerun: ReplayStep | null;
  diverged: boolean;
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
