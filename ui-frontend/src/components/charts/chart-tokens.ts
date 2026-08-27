/* ------------------------------------------------------------------ *
 * Chart constants + pure formatters, shared by every recharts surface.
 *
 * JSX-free on purpose so `react-refresh/only-export-components` stays happy
 * for the .tsx half (chart-kit.tsx).
 *
 * Ported from the Enterprise /next analytics chart layer so the two products
 * draw the same way. Everything resolves to a theme token, so charts follow
 * the skin and the light/dark theme without any per-chart logic.
 * ------------------------------------------------------------------ */

/** Reserved status palette. Never reused for a categorical series. */
export const STATUS_FILL = {
  completed: "var(--fa-success)",
  failed: "var(--destructive)",
  other: "var(--muted-foreground)",
} as const;

/**
 * The categorical series order — FIXED, assigned by entity, never cycled or
 * repainted when a filter changes the series count. A 6th series is "Other",
 * never a generated hue.
 */
export const CATEGORICAL = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
] as const;

/** Recessive, hairline, solid — the grid is never the loudest mark. */
export const AXIS_PROPS = {
  stroke: "var(--fa-grid-line)",
  tick: { fill: "var(--muted-foreground)", fontSize: 10 },
} as const;

export const GRID_STROKE = "var(--fa-grid-line)";

/** Series colour for the nth entity, with "Other" for anything past the ramp. */
export function seriesColor(index: number): string {
  return index < CATEGORICAL.length
    ? CATEGORICAL[index]
    : "var(--muted-foreground)";
}

export const fmtUsd = (n: number) =>
  n >= 0.01 ? `$${n.toFixed(2)}` : `$${n.toFixed(4)}`;
