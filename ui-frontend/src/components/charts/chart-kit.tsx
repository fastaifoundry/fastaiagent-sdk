import { cn } from "@/lib/utils";

/**
 * The one chart wrapper for the Local UI.
 *
 * Everything chart-shaped goes through these primitives so there is a single
 * look, a single tooltip, and a single place to change axis styling.
 *
 * Colour is applied by the *job* the data does — categorical for identity, the
 * reserved status palette for state. Status never doubles as "series 4", and it
 * always ships with a label, never colour alone.
 *
 * Constants and formatters live in the sibling `chart-tokens.ts`.
 */

/** A titled chart/figure frame. */
export function Panel({
  title,
  subtitle,
  action,
  children,
  className,
}: {
  title: string;
  subtitle?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section
      className={cn("rounded-xl border border-border bg-card p-4", className)}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="fa-section-label">{title}</h3>
          {subtitle && (
            <p className="mt-1 text-[11.5px] text-muted-foreground">
              {subtitle}
            </p>
          )}
        </div>
        {action}
      </div>
      <div className="mt-3">{children}</div>
    </section>
  );
}

/** One tooltip shell for every chart. Values wear ink tokens, not series colour. */
export function ChartTooltip({
  active,
  payload,
  label,
  valueFormatter,
  labelFormatter,
}: {
  active?: boolean;
  payload?: { name?: string; value?: number | string; color?: string }[];
  label?: string | number;
  valueFormatter?: (v: number) => string;
  labelFormatter?: (v: string | number) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-border bg-card px-2.5 py-1.5 shadow-md">
      {label != null && (
        <div className="mb-1 font-mono text-[10.5px] text-muted-foreground">
          {labelFormatter ? labelFormatter(label) : label}
        </div>
      )}
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-1.5 font-mono text-[11px]">
          <span
            className="h-2 w-2 rounded-sm"
            style={{ background: p.color }}
          />
          <span className="text-muted-foreground">{p.name}</span>
          <span className="ml-auto pl-3 font-semibold tabular-nums text-foreground">
            {typeof p.value === "number"
              ? valueFormatter
                ? valueFormatter(p.value)
                : p.value.toLocaleString()
              : String(p.value ?? "")}
          </span>
        </div>
      ))}
    </div>
  );
}
