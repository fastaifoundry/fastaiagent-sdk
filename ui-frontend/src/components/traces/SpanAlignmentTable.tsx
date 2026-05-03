import { Fragment, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { formatDurationMs } from "@/lib/format";
import type {
  CompareAlignmentRow,
  CompareMatchKind,
  CompareTraceHalf,
  SpanRow,
} from "@/lib/types";
import { SpanDiffPanel } from "./SpanDiffPanel";

interface Props {
  rows: CompareAlignmentRow[];
  traceA: CompareTraceHalf;
  traceB: CompareTraceHalf;
}

const MATCH_LABELS: Record<CompareMatchKind, string> = {
  same: "same",
  slower: "slower",
  faster: "faster",
  different_output: "different output",
  new_in_a: "new in A",
  new_in_b: "new in B",
};

const MATCH_STYLES: Record<CompareMatchKind, string> = {
  same: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20",
  faster: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20",
  slower: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20",
  different_output:
    "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20",
  new_in_a: "bg-sky-500/10 text-sky-600 dark:text-sky-400 border-sky-500/20",
  new_in_b: "bg-sky-500/10 text-sky-600 dark:text-sky-400 border-sky-500/20",
};

function MatchBadge({ kind }: { kind: CompareMatchKind }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md border px-2 py-0.5 text-[10px] font-mono font-medium uppercase tracking-wider",
        MATCH_STYLES[kind]
      )}
    >
      {MATCH_LABELS[kind]}
    </span>
  );
}

function findSpan(half: CompareTraceHalf, span_id: string | undefined): SpanRow | null {
  if (!span_id) return null;
  return half.spans.find((s) => s.span_id === span_id) ?? null;
}

/**
 * Aligned span table for the comparison view. Each row pairs a span from
 * trace A with a span from trace B (when matched), or shows the lone span
 * when only one side has it. Click a row to expand the input/output/
 * attributes diff via :class:`SpanDiffPanel`.
 */
export function SpanAlignmentTable({ rows, traceA, traceB }: Props) {
  const [open, setOpen] = useState<Set<number>>(new Set());

  const toggle = (idx: number) => {
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  return (
    <div data-testid="span-alignment-table" className="rounded-md border bg-card">
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead className="w-[40px]"></TableHead>
            <TableHead className="w-[48px] text-right">#</TableHead>
            <TableHead>Trace A span</TableHead>
            <TableHead className="w-[100px] text-right">A duration</TableHead>
            <TableHead>Trace B span</TableHead>
            <TableHead className="w-[100px] text-right">B duration</TableHead>
            <TableHead className="w-[140px]">Match</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => {
            const isOpen = open.has(row.index);
            const fullA = findSpan(traceA, row.span_a?.span_id);
            const fullB = findSpan(traceB, row.span_b?.span_id);
            return (
              <Fragment key={row.index}>
                <TableRow
                  className="cursor-pointer"
                  onClick={() => toggle(row.index)}
                >
                  <TableCell>
                    <button
                      type="button"
                      aria-label={
                        isOpen ? "Collapse span diff" : "Expand span diff"
                      }
                      className="inline-flex h-5 w-5 items-center justify-center rounded hover:bg-muted"
                      onClick={(e) => {
                        e.stopPropagation();
                        toggle(row.index);
                      }}
                    >
                      {isOpen ? (
                        <ChevronDown className="h-3 w-3" />
                      ) : (
                        <ChevronRight className="h-3 w-3" />
                      )}
                    </button>
                  </TableCell>
                  <TableCell className="text-right font-mono text-xs text-muted-foreground">
                    {row.index}
                  </TableCell>
                  <TableCell className="font-mono text-xs">
                    {row.span_a?.name ?? <span className="text-muted-foreground">—</span>}
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums text-xs">
                    {formatDurationMs(row.span_a?.duration_ms ?? null)}
                  </TableCell>
                  <TableCell className="font-mono text-xs">
                    {row.span_b?.name ?? <span className="text-muted-foreground">—</span>}
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums text-xs">
                    {formatDurationMs(row.span_b?.duration_ms ?? null)}
                  </TableCell>
                  <TableCell>
                    <MatchBadge kind={row.match} />
                    {row.delta_ms != null && row.delta_ms !== 0 && (
                      <span className="ml-2 font-mono text-[11px] text-muted-foreground">
                        {row.delta_ms > 0 ? "+" : ""}
                        {row.delta_ms}ms
                      </span>
                    )}
                  </TableCell>
                </TableRow>
                {isOpen && (
                  <TableRow className="hover:bg-transparent">
                    <TableCell colSpan={7} className="bg-muted/30 p-4">
                      <SpanDiffPanel spanA={fullA} spanB={fullB} />
                    </TableCell>
                  </TableRow>
                )}
              </Fragment>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
