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
import type { EvalTrendPoint } from "@/lib/types";

interface Props {
  points: EvalTrendPoint[];
}

export function QualityTrendChart({ points }: Props) {
  const data = useMemo(
    () =>
      points.map((p) => ({
        ts: new Date(p.started_at).getTime(),
        label: new Date(p.started_at).toLocaleString(),
        rate: Math.round((p.pass_rate ?? 0) * 100),
        dataset: p.dataset_name,
      })),
    [points]
  );

  if (data.length === 0) {
    return (
      <p className="py-8 text-center text-xs text-muted-foreground">
        Not enough runs to draw a trend yet.
      </p>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={180}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 12 }}>
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
          contentStyle={{
            background: "var(--color-card)",
            border: "1px solid var(--color-border)",
            borderRadius: 6,
            fontSize: 12,
          }}
          labelFormatter={(value) => new Date(value).toLocaleString()}
          formatter={(value) => [`${value}%`, "Pass rate"] as [string, string]}
        />
        <Line
          type="monotone"
          dataKey="rate"
          stroke="var(--color-primary)"
          strokeWidth={2}
          dot={{ r: 3 }}
          activeDot={{ r: 5 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
