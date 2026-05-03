import { ArrowDown, ArrowRight, ArrowUp } from "lucide-react";
import { cn } from "@/lib/utils";
import { formatCost, formatDurationMs, formatTokens } from "@/lib/format";
import type { CompareSummary } from "@/lib/types";

interface Props {
  summary: CompareSummary;
}

type Direction = "lower-better" | "higher-better" | "neutral";

function deltaClass(value: number | null, direction: Direction): string {
  if (value == null || value === 0) return "text-muted-foreground";
  if (direction === "neutral") return "text-muted-foreground";
  const improved =
    direction === "lower-better" ? value < 0 : value > 0;
  return improved
    ? "text-emerald-600 dark:text-emerald-400"
    : "text-red-600 dark:text-red-400";
}

function DeltaIcon({ value }: { value: number | null }) {
  if (value == null || value === 0) {
    return <ArrowRight className="h-3 w-3" />;
  }
  return value > 0 ? (
    <ArrowUp className="h-3 w-3" />
  ) : (
    <ArrowDown className="h-3 w-3" />
  );
}

function Cell({
  label,
  value,
  delta,
  format,
  direction,
}: {
  label: string;
  value: string;
  delta: number | null;
  format: (v: number) => string;
  direction: Direction;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        {label}
      </span>
      <span className="text-sm font-mono tabular-nums">{value}</span>
      <span
        className={cn(
          "inline-flex items-center gap-1 text-[11px] font-mono tabular-nums",
          deltaClass(delta, direction)
        )}
      >
        <DeltaIcon value={delta} />
        {delta == null ? "—" : (delta > 0 ? "+" : "") + format(delta)}
      </span>
    </div>
  );
}

/**
 * Side-by-side KPI delta cards for trace comparison.
 *
 * Direction semantics: lower duration/tokens/cost are "better" (green).
 * spans_delta is neutral — more spans isn't intrinsically worse.
 */
export function TraceCompareSummary({ summary }: Props) {
  return (
    <div
      data-testid="trace-compare-summary"
      className="grid grid-cols-2 gap-4 rounded-md border bg-card px-4 py-3 sm:grid-cols-4"
    >
      <Cell
        label="Duration"
        value={
          summary.duration_delta_ms == null
            ? "—"
            : formatDurationMs(Math.abs(summary.duration_delta_ms)) + " diff"
        }
        delta={summary.duration_delta_ms}
        format={(v) => formatDurationMs(Math.abs(v))}
        direction="lower-better"
      />
      <Cell
        label="Tokens"
        value={
          summary.tokens_delta == null
            ? "—"
            : formatTokens(Math.abs(summary.tokens_delta)) + " diff"
        }
        delta={summary.tokens_delta}
        format={(v) => formatTokens(Math.abs(v))}
        direction="lower-better"
      />
      <Cell
        label="Cost"
        value={
          summary.cost_delta_usd == null
            ? "—"
            : formatCost(Math.abs(summary.cost_delta_usd)) + " diff"
        }
        delta={summary.cost_delta_usd}
        format={(v) => formatCost(Math.abs(v))}
        direction="lower-better"
      />
      <Cell
        label="Spans"
        value={
          summary.spans_delta === 0
            ? "—"
            : `${Math.abs(summary.spans_delta)} diff`
        }
        delta={summary.spans_delta}
        format={(v) => String(Math.abs(v))}
        direction="neutral"
      />
    </div>
  );
}
