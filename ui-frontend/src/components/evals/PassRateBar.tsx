import { cn } from "@/lib/utils";

interface Props {
  /** A pass rate in 0..1; `null` means no verdict applied, shown as "—". */
  passRate: number | null | undefined;
  passCount?: number | null;
  failCount?: number | null;
  /** Hover text, e.g. the checks rate next to a cases rate. */
  title?: string;
}

export function PassRateBar({ passRate, passCount, failCount, title }: Props) {
  const known = typeof passRate === "number" && Number.isFinite(passRate);
  const rate = known ? passRate : 0;
  const pct = Math.round(rate * 100);
  const color =
    rate >= 0.9
      ? "bg-fa-success"
      : rate >= 0.7
      ? "bg-fa-warning"
      : "bg-destructive";

  return (
    <div className="flex items-center gap-2" title={title}>
      <div className="flex h-2 w-24 overflow-hidden rounded-full bg-muted">
        {known && (
          <div
            className={cn("h-full", color)}
            style={{ width: `${pct}%` }}
          />
        )}
      </div>
      <span className="font-mono text-xs tabular-nums text-muted-foreground">
        {known ? `${pct}%` : "—"}
      </span>
      {(passCount != null || failCount != null) && (
        <span className="font-mono text-[11px] text-muted-foreground/70">
          ({passCount ?? 0}/{(passCount ?? 0) + (failCount ?? 0)})
        </span>
      )}
    </div>
  );
}
