import { useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { PageHeader } from "@/components/layout/PageHeader";
import { StatCard } from "@/components/shared/StatCard";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { CostBreakdownTable } from "@/components/analytics/CostBreakdownTable";
import { ChartTooltip, Panel } from "@/components/charts/chart-kit";
import {
  AXIS_PROPS,
  GRID_STROKE,
  STATUS_FILL,
  fmtUsd,
} from "@/components/charts/chart-tokens";
import { useCostBreakdown } from "@/hooks/use-cost-breakdown";
import type { CostByModelRow } from "@/lib/types";
import { useAnalytics } from "@/hooks/use-analytics";
import { formatCost, formatDurationMs } from "@/lib/format";
import { cn } from "@/lib/utils";

const WINDOW_CHOICES: { label: string; hours: number; granularity: "hour" | "day" }[] = [
  { label: "24h", hours: 24, granularity: "hour" },
  { label: "7d", hours: 24 * 7, granularity: "hour" },
  { label: "30d", hours: 24 * 30, granularity: "day" },
];

type CostPeriod = "1d" | "7d" | "30d" | "all";

const CHART_MARGIN = { top: 6, right: 20, bottom: 4, left: 12 };

/** Shared time-axis config — every timeline on this page reads the same. */
const TIME_AXIS = {
  type: "number" as const,
  scale: "time" as const,
  domain: ["dataMin", "dataMax"] as [string, string],
  tickFormatter: (v: number) =>
    new Date(v).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    }),
  ...AXIS_PROPS,
};

function periodFromChoice(label: string): CostPeriod {
  if (label === "24h") return "1d";
  if (label === "7d") return "7d";
  if (label === "30d") return "30d";
  return "all";
}

export function AnalyticsPage() {
  const [choice, setChoice] = useState(WINDOW_CHOICES[1]);
  const analytics = useAnalytics(choice.hours, choice.granularity);
  const [costChainName, setCostChainName] = useState("");

  const chartData = useMemo(() => {
    const rows = analytics.data?.points ?? [];
    return rows.map((p) => ({
      bucket: new Date(p.bucket).getTime(),
      label: new Date(p.bucket).toLocaleString(),
      p50: p.p50_ms,
      p95: p.p95_ms,
      p99: p.p99_ms,
      cost: p.cost_usd,
      errorRate: Math.round(p.error_rate * 100),
      traces: p.trace_count,
      // Split the bucket so failures stack visibly on top of successes.
      // The API reports totals and errors; "ok" is the remainder.
      failed: p.error_count,
      ok: Math.max(0, p.trace_count - p.error_count),
    }));
  }, [analytics.data]);

  // The cost dashboard below already fetches this breakdown; reusing the same
  // query key means the two panels share one request, not two.
  const costPeriod = periodFromChoice(choice.label);
  const modelCosts = useCostBreakdown({ groupBy: "model", period: costPeriod });

  const modelRows = useMemo(() => {
    const rows = (modelCosts.data?.rows ?? []) as CostByModelRow[];
    // Largest first — recharts renders the first datum at the TOP in a
    // vertical layout, so this reads as a ranked list.
    return [...rows].sort((a, b) => b.cost_usd - a.cost_usd).slice(0, 8);
  }, [modelCosts.data]);

  const tokenSplit = useMemo(() => {
    const rows = (modelCosts.data?.rows ?? []) as CostByModelRow[];
    const prompt = rows.reduce((n, r) => n + (r.input_tokens ?? 0), 0);
    const completion = rows.reduce((n, r) => n + (r.output_tokens ?? 0), 0);
    const total = prompt + completion;
    return { prompt, completion, total, promptPct: total ? (prompt / total) * 100 : 0 };
  }, [modelCosts.data]);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Analytics"
        description="Latency, cost, and error trends across all traces."
      >
        <div className="flex items-center gap-1 rounded-md border bg-card p-0.5">
          {WINDOW_CHOICES.map((c) => (
            <button
              key={c.label}
              type="button"
              onClick={() => setChoice(c)}
              className={cn(
                "rounded px-2 py-1 text-xs font-mono font-medium transition-colors",
                choice.label === c.label
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
              )}
            >
              {c.label}
            </button>
          ))}
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => analytics.refetch()}
          disabled={analytics.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${analytics.isFetching ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </PageHeader>

      {analytics.isLoading ? (
        <TableSkeleton rows={5} />
      ) : !analytics.data ? (
        <EmptyState title="No analytics yet" />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-4">
            <StatCard
              label="Traces"
              value={analytics.data.summary.trace_count.toLocaleString()}
            />
            <StatCard
              label="Success rate"
              value={`${(
                (1 - analytics.data.summary.error_rate) *
                100
              ).toFixed(1)}%`}
            />
            <StatCard
              label="Errors"
              value={`${analytics.data.summary.error_count} (${Math.round(
                analytics.data.summary.error_rate * 100
              )}%)`}
            />
            <StatCard
              label="Total cost"
              value={formatCost(analytics.data.summary.total_cost_usd)}
            />
          </div>

          {/* Percentiles get their own row. P99 was already computed by the
              API and thrown away by the UI — which is exactly the number that
              catches a slow tail hiding behind a healthy median. */}
          <div className="grid grid-cols-3 gap-4">
            <StatCard
              label="Latency P50"
              value={formatDurationMs(analytics.data.summary.p50_ms)}
            />
            <StatCard
              label="Latency P95"
              value={formatDurationMs(analytics.data.summary.p95_ms)}
            />
            <StatCard
              label="Latency P99"
              value={formatDurationMs(analytics.data.summary.p99_ms)}
            />
          </div>

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <Panel
              title="Latency percentiles"
              subtitle="p50 · p95 · p99 over the selected window"
            >
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={chartData} margin={CHART_MARGIN}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke={GRID_STROKE}
                    vertical={false}
                  />
                  <XAxis dataKey="bucket" {...TIME_AXIS} />
                  <YAxis
                    tickFormatter={(v) => formatDurationMs(Number(v))}
                    width={72}
                    {...AXIS_PROPS}
                  />
                  <Tooltip
                    cursor={{ stroke: "var(--primary)", strokeWidth: 1 }}
                    content={
                      <ChartTooltip
                        labelFormatter={(v) => new Date(v).toLocaleString()}
                        valueFormatter={(v) => formatDurationMs(v)}
                      />
                    }
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {/* Percentiles are a severity ramp, not identity — the
                      reserved status palette is the right scale here. */}
                  <Line type="monotone" dataKey="p50" stroke={STATUS_FILL.completed} name="p50" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="p95" stroke="var(--fa-warning)" name="p95" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="p99" stroke={STATUS_FILL.failed} name="p99" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Panel>

            <Panel title="Cost over time" subtitle="USD spend per bucket">
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={chartData} margin={CHART_MARGIN}>
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} vertical={false} />
                  <XAxis dataKey="bucket" {...TIME_AXIS} />
                  <YAxis
                    tickFormatter={(v) => formatCost(Number(v))}
                    width={72}
                    {...AXIS_PROPS}
                  />
                  <Tooltip
                    cursor={{ stroke: "var(--primary)", strokeWidth: 1 }}
                    content={
                      <ChartTooltip
                        labelFormatter={(v) => new Date(v).toLocaleString()}
                        valueFormatter={(v) => formatCost(v)}
                      />
                    }
                  />
                  <Line type="monotone" dataKey="cost" stroke="var(--chart-1)" name="Cost" strokeWidth={2} dot={{ r: 2 }} />
                </LineChart>
              </ResponsiveContainer>
            </Panel>

            <Panel title="Error rate" subtitle="Share of traces ending in error">
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={chartData} margin={CHART_MARGIN}>
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} vertical={false} />
                  <XAxis dataKey="bucket" {...TIME_AXIS} />
                  <YAxis
                    tickFormatter={(v) => `${v}%`}
                    width={48}
                    domain={[0, 100]}
                    {...AXIS_PROPS}
                  />
                  <Tooltip
                    cursor={{ stroke: "var(--primary)", strokeWidth: 1 }}
                    content={
                      <ChartTooltip
                        labelFormatter={(v) => new Date(v).toLocaleString()}
                        valueFormatter={(v) => `${v}%`}
                      />
                    }
                  />
                  <Line type="monotone" dataKey="errorRate" stroke={STATUS_FILL.failed} name="Error rate" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Panel>

            {/* Was a single "Trace volume" line, which hid the failure mix:
                a flat total can be 100% healthy or 40% failing and look the
                same. Stacking failures on top keeps them visible at any
                height, and status always ships with a legend — never colour
                alone. */}
            <Panel
              title="Volume by status"
              subtitle={`Traces per ${choice.granularity}, failures stacked on top`}
            >
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={chartData} margin={CHART_MARGIN}>
                  <CartesianGrid stroke={GRID_STROKE} vertical={false} />
                  <XAxis dataKey="bucket" {...TIME_AXIS} />
                  <YAxis width={48} allowDecimals={false} {...AXIS_PROPS} />
                  <Tooltip
                    cursor={{ fill: "var(--muted)", opacity: 0.35 }}
                    content={
                      <ChartTooltip
                        labelFormatter={(v) => new Date(v).toLocaleString()}
                      />
                    }
                  />
                  <Legend wrapperStyle={{ fontSize: 11, paddingTop: 6 }} iconType="square" iconSize={8} />
                  {/* A surface-coloured hairline separates the segments. */}
                  <Bar dataKey="ok" name="ok" stackId="s" fill={STATUS_FILL.completed}
                       maxBarSize={24} stroke="var(--card)" strokeWidth={2} />
                  <Bar dataKey="failed" name="failed" stackId="s" fill={STATUS_FILL.failed}
                       maxBarSize={24} radius={[4, 4, 0, 0]} stroke="var(--card)" strokeWidth={2} />
                </BarChart>
              </ResponsiveContainer>
            </Panel>
          </div>

          {(modelRows.length > 0 || tokenSplit.total > 0) && (
            <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-2">
              <Panel
                title="Top models by cost"
                subtitle="Estimated from recorded tokens and published pricing"
              >
                {modelRows.length === 0 ? (
                  <p className="py-6 text-center text-[12.5px] text-muted-foreground">
                    No priced model calls in this period.
                  </p>
                ) : (
                  // Magnitude → one hue; one series → no legend; every bar is
                  // direct-labelled, so the panel also reads as a table.
                  <ResponsiveContainer width="100%" height={Math.max(140, modelRows.length * 30)}>
                    <BarChart data={modelRows} layout="vertical" margin={{ top: 0, right: 64, bottom: 0, left: 0 }}>
                      <XAxis type="number" hide />
                      <YAxis
                        type="category"
                        dataKey="model"
                        width={150}
                        tickLine={false}
                        axisLine={false}
                        tickFormatter={(m: string) => (m.length > 20 ? `${m.slice(0, 19)}…` : m)}
                        {...AXIS_PROPS}
                      />
                      <Tooltip
                        cursor={{ fill: "var(--muted)", opacity: 0.35 }}
                        content={<ChartTooltip valueFormatter={fmtUsd} />}
                      />
                      <Bar
                        dataKey="cost_usd"
                        name="cost"
                        fill="var(--chart-1)"
                        maxBarSize={18}
                        radius={[0, 4, 4, 0]}
                        label={{
                          position: "right",
                          formatter: (v: unknown) => (typeof v === "number" ? fmtUsd(v) : ""),
                          fill: "var(--muted-foreground)",
                          fontSize: 10,
                        }}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </Panel>

              <Panel
                title="Token split"
                subtitle="Prompt vs completion tokens across all models"
              >
                {tokenSplit.total === 0 ? (
                  <p className="py-6 text-center text-[12.5px] text-muted-foreground">
                    No token usage recorded in this period.
                  </p>
                ) : (
                  <>
                    {/* Part-to-whole of two → one stacked bar, direct-labelled below. */}
                    <div className="flex h-7 w-full overflow-hidden rounded-md">
                      <div style={{ width: `${tokenSplit.promptPct}%`, background: "var(--chart-1)" }} />
                      <div style={{ width: "2px", background: "var(--card)" }} />
                      <div style={{ flex: 1, background: "var(--chart-2)" }} />
                    </div>
                    <div className="mt-3 space-y-1.5">
                      {[
                        { name: "Prompt", tok: tokenSplit.prompt, color: "var(--chart-1)" },
                        { name: "Completion", tok: tokenSplit.completion, color: "var(--chart-2)" },
                      ].map((r) => (
                        <div key={r.name} className="flex items-center gap-2 text-[12px]">
                          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: r.color }} />
                          <span className="text-muted-foreground">{r.name}</span>
                          <span className="ml-auto font-mono tabular-nums">
                            {r.tok.toLocaleString()}
                          </span>
                          <span className="w-14 text-right font-mono text-[11px] tabular-nums text-muted-foreground">
                            {Math.round((r.tok / tokenSplit.total) * 100)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </Panel>
            </div>
          )}

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Top 5 slowest agents</CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow className="hover:bg-transparent">
                      <TableHead>Agent</TableHead>
                      <TableHead className="text-right">Runs</TableHead>
                      <TableHead className="text-right">Avg latency</TableHead>
                      <TableHead className="text-right">Errors</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {analytics.data.top_slowest_agents.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={4} className="text-center text-xs text-muted-foreground">
                          Not enough data.
                        </TableCell>
                      </TableRow>
                    ) : (
                      analytics.data.top_slowest_agents.map((a) => (
                        <TableRow key={a.agent_name}>
                          <TableCell className="font-mono text-sm">{a.agent_name}</TableCell>
                          <TableCell className="text-right font-mono tabular-nums">{a.run_count}</TableCell>
                          <TableCell className="text-right font-mono tabular-nums">
                            {formatDurationMs(a.avg_latency_ms ?? null)}
                          </TableCell>
                          <TableCell className="text-right font-mono tabular-nums text-muted-foreground">
                            {a.error_count}
                          </TableCell>
                        </TableRow>
                      ))
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Top 5 priciest agents</CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow className="hover:bg-transparent">
                      <TableHead>Agent</TableHead>
                      <TableHead className="text-right">Runs</TableHead>
                      <TableHead className="text-right">Total cost</TableHead>
                      <TableHead className="text-right">Avg cost</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {analytics.data.top_priciest_agents.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={4} className="text-center text-xs text-muted-foreground">
                          No priced runs.
                        </TableCell>
                      </TableRow>
                    ) : (
                      analytics.data.top_priciest_agents.map((a) => (
                        <TableRow key={a.agent_name}>
                          <TableCell className="font-mono text-sm">{a.agent_name}</TableCell>
                          <TableCell className="text-right font-mono tabular-nums">{a.run_count}</TableCell>
                          <TableCell className="text-right font-mono tabular-nums">
                            {formatCost(a.total_cost_usd)}
                          </TableCell>
                          <TableCell className="text-right font-mono tabular-nums text-muted-foreground">
                            {formatCost(a.avg_cost_usd ?? null)}
                          </TableCell>
                        </TableRow>
                      ))
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </div>

          <div data-testid="cost-breakdown-section">
            <h2 className="mb-3 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              // COST BREAKDOWN
            </h2>
            <Tabs defaultValue="model">
              <TabsList>
                <TabsTrigger value="model">By model</TabsTrigger>
                <TabsTrigger value="agent">By agent</TabsTrigger>
                <TabsTrigger value="node">By node</TabsTrigger>
              </TabsList>
              <TabsContent value="model" className="mt-3">
                <CostBreakdownTable
                  groupBy="model"
                  period={periodFromChoice(choice.label)}
                />
              </TabsContent>
              <TabsContent value="agent" className="mt-3">
                <CostBreakdownTable
                  groupBy="agent"
                  period={periodFromChoice(choice.label)}
                />
              </TabsContent>
              <TabsContent value="node" className="mt-3 space-y-3">
                <Input
                  type="text"
                  placeholder="Enter chain name (required) — e.g. support-flow"
                  value={costChainName}
                  onChange={(e) => setCostChainName(e.target.value)}
                  className="max-w-sm font-mono text-sm"
                  aria-label="Chain name"
                />
                {costChainName ? (
                  <CostBreakdownTable
                    groupBy="node"
                    period={periodFromChoice(choice.label)}
                    chainName={costChainName}
                  />
                ) : (
                  <p className="text-xs text-muted-foreground">
                    Enter a chain name above to see per-node cost breakdown.
                  </p>
                )}
              </TabsContent>
            </Tabs>
          </div>

          <p className="rounded-lg bg-muted/50 p-3 font-mono text-xs text-muted-foreground">
            Cost figures are estimates derived from recorded token counts and
            published API pricing — local models (e.g. Ollama) show $0.00.
            Actual billing may differ under your provider agreement.
          </p>
        </>
      )}
    </div>
  );
}
