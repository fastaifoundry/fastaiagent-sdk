import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { seriesColor } from "@/components/charts/chart-tokens";
import { buildTrend, type TrendRow } from "@/lib/evalDisplay";
import type { EvalTrendPoint } from "@/lib/types";

interface Props {
  points: EvalTrendPoint[];
  /** The points include dataset checks and AutoLLM runs (`include=all`). */
  includesAll?: boolean;
}

function TrendTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload?: TrendRow; color?: string }[];
}) {
  const row = active ? payload?.[0]?.payload : undefined;
  if (!row) return null;
  return (
    <div className="rounded-lg border border-border bg-card px-2.5 py-1.5 font-mono text-[11px] shadow-md">
      <div className="mb-1 text-[10.5px] text-muted-foreground">
        {new Date(row.ts).toLocaleString()}
      </div>
      <div className="flex items-center gap-1.5">
        <span className="h-2 w-2 rounded-sm" style={{ background: payload?.[0]?.color }} />
        <span className="text-foreground">{row.series}</span>
      </div>
      {row.runName && row.runName !== row.series && (
        <div className="text-muted-foreground">{row.runName}</div>
      )}
      <div className="mt-1">
        Cases <span className="font-semibold tabular-nums">{row.rate}%</span>
      </div>
      <div className="text-muted-foreground">
        Checks {row.checks != null ? `${row.checks}%` : "—"}
      </div>
      {row.gate && <div className="text-muted-foreground">Gate {row.gate}</div>}
    </div>
  );
}

/**
 * Cases pass rate over time, one line per dataset (or pytest session), the
 * five with the most runs. A run with no case verdict is left out, not drawn
 * as 0%.
 */
export function QualityTrendChart({ points, includesAll = false }: Props) {
  const { rows, series } = useMemo(() => buildTrend(points), [points]);

  if (rows.length === 0) {
    return (
      <p className="py-8 text-center text-xs text-muted-foreground">
        Not enough runs to draw a trend yet.
      </p>
    );
  }

  return (
    <div>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 4, left: 12 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-border" vertical={false} />
          <XAxis
            dataKey="ts"
            type="number"
            scale="time"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(value) =>
              new Date(value).toLocaleDateString(undefined, {
                month: "short",
                day: "numeric",
              })
            }
            stroke="currentColor"
            className="text-muted-foreground text-xs"
          />
          <YAxis
            domain={[0, 100]}
            tickFormatter={(v) => `${v}%`}
            stroke="currentColor"
            className="text-muted-foreground text-xs"
            width={48}
          />
          <Tooltip
            cursor={{ stroke: "var(--color-primary)", strokeWidth: 1 }}
            content={<TrendTooltip />}
          />
          {series.map((name, i) => (
            <Line
              key={name}
              name={name}
              type="monotone"
              dataKey={(r: TrendRow) => (r.series === name ? r.rate : undefined)}
              stroke={seriesColor(i)}
              strokeWidth={2}
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
              connectNulls
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      {series.length > 1 && (
        <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted-foreground">
          {series.map((name, i) => (
            <span key={name} className="inline-flex items-center gap-1.5 font-mono">
              <span className="h-2 w-2 rounded-sm" style={{ background: seriesColor(i) }} />
              {name}
            </span>
          ))}
        </div>
      )}
      <p className="mt-1 text-[11px] text-muted-foreground" data-testid="trend-caption">
        {includesAll
          ? "Cases pass rate. Includes dataset checks and AutoLLM candidate runs, which do not measure the agent. Invalid (outage) runs are left out."
          : "Cases pass rate. Dataset checks, AutoLLM candidate runs and invalid (outage) runs are left out."}
      </p>
    </div>
  );
}
