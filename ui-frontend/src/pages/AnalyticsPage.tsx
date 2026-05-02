import { useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";
import {
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
import { useAnalytics } from "@/hooks/use-analytics";
import { formatCost, formatDurationMs } from "@/lib/format";
import { cn } from "@/lib/utils";

const WINDOW_CHOICES: { label: string; hours: number; granularity: "hour" | "day" }[] = [
  { label: "24h", hours: 24, granularity: "hour" },
  { label: "7d", hours: 24 * 7, granularity: "hour" },
  { label: "30d", hours: 24 * 30, granularity: "day" },
];

type CostPeriod = "1d" | "7d" | "30d" | "all";

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
    }));
  }, [analytics.data]);

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
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-5">
            <StatCard
              label="Traces"
              value={analytics.data.summary.trace_count.toLocaleString()}
            />
            <StatCard
              label="Errors"
              value={`${analytics.data.summary.error_count} (${Math.round(
                analytics.data.summary.error_rate * 100
              )}%)`}
            />
            <StatCard
              label="P50"
              value={formatDurationMs(analytics.data.summary.p50_ms)}
            />
            <StatCard
              label="P95"
              value={formatDurationMs(analytics.data.summary.p95_ms)}
            />
            <StatCard
              label="Total cost"
              value={formatCost(analytics.data.summary.total_cost_usd)}
            />
          </div>

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Latency percentiles</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={chartData} margin={{ top: 6, right: 20, bottom: 4, left: 12 }}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      className="stroke-border"
                      vertical={false}
                    />
                    <XAxis
                      dataKey="bucket"
                      type="number"
                      scale="time"
                      domain={["dataMin", "dataMax"]}
                      tickFormatter={(v) => new Date(v).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
                      className="text-muted-foreground text-xs"
                      stroke="currentColor"
                    />
                    <YAxis
                      tickFormatter={(v) => formatDurationMs(Number(v))}
                      width={72}
                      className="text-muted-foreground text-xs"
                      stroke="currentColor"
                    />
                    <Tooltip
                      cursor={{ stroke: "var(--color-primary)", strokeWidth: 1 }}
                      contentStyle={{
                        background: "var(--color-card)",
                        border: "1px solid var(--color-border)",
                        borderRadius: 6,
                        fontSize: 12,
                      }}
                      labelFormatter={(v) => new Date(v).toLocaleString()}
                      formatter={(value) => [formatDurationMs(Number(value)), ""]}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Line type="monotone" dataKey="p50" stroke="var(--color-fa-success)" name="p50" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="p95" stroke="var(--color-fa-warning)" name="p95" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="p99" stroke="var(--color-destructive)" name="p99" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Cost over time</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={chartData} margin={{ top: 6, right: 20, bottom: 4, left: 12 }}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-border" vertical={false} />
                    <XAxis
                      dataKey="bucket"
                      type="number"
                      scale="time"
                      domain={["dataMin", "dataMax"]}
                      tickFormatter={(v) => new Date(v).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
                      className="text-muted-foreground text-xs"
                      stroke="currentColor"
                    />
                    <YAxis
                      tickFormatter={(v) => formatCost(Number(v))}
                      width={72}
                      className="text-muted-foreground text-xs"
                      stroke="currentColor"
                    />
                    <Tooltip
                      cursor={{ stroke: "var(--color-primary)", strokeWidth: 1 }}
                      contentStyle={{
                        background: "var(--color-card)",
                        border: "1px solid var(--color-border)",
                        borderRadius: 6,
                        fontSize: 12,
                      }}
                      labelFormatter={(v) => new Date(v).toLocaleString()}
                      formatter={(value) => [formatCost(Number(value)), "Cost"]}
                    />
                    <Line type="monotone" dataKey="cost" stroke="var(--color-primary)" name="cost" strokeWidth={2} dot={{ r: 2 }} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Error rate</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={chartData} margin={{ top: 6, right: 20, bottom: 4, left: 12 }}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-border" vertical={false} />
                    <XAxis
                      dataKey="bucket"
                      type="number"
                      scale="time"
                      domain={["dataMin", "dataMax"]}
                      tickFormatter={(v) => new Date(v).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
                      className="text-muted-foreground text-xs"
                      stroke="currentColor"
                    />
                    <YAxis
                      tickFormatter={(v) => `${v}%`}
                      width={48}
                      className="text-muted-foreground text-xs"
                      stroke="currentColor"
                      domain={[0, 100]}
                    />
                    <Tooltip
                      cursor={{ stroke: "var(--color-primary)", strokeWidth: 1 }}
                      contentStyle={{
                        background: "var(--color-card)",
                        border: "1px solid var(--color-border)",
                        borderRadius: 6,
                        fontSize: 12,
                      }}
                      labelFormatter={(v) => new Date(v).toLocaleString()}
                      formatter={(value) => [`${value}%`, "Error rate"]}
                    />
                    <Line type="monotone" dataKey="errorRate" stroke="var(--color-destructive)" name="errorRate" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Trace volume</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={chartData} margin={{ top: 6, right: 20, bottom: 4, left: 12 }}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-border" vertical={false} />
                    <XAxis
                      dataKey="bucket"
                      type="number"
                      scale="time"
                      domain={["dataMin", "dataMax"]}
                      tickFormatter={(v) => new Date(v).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
                      className="text-muted-foreground text-xs"
                      stroke="currentColor"
                    />
                    <YAxis width={48} className="text-muted-foreground text-xs" stroke="currentColor" />
                    <Tooltip
                      cursor={{ stroke: "var(--color-primary)", strokeWidth: 1 }}
                      contentStyle={{
                        background: "var(--color-card)",
                        border: "1px solid var(--color-border)",
                        borderRadius: 6,
                        fontSize: 12,
                      }}
                      labelFormatter={(v) => new Date(v).toLocaleString()}
                    />
                    <Line type="monotone" dataKey="traces" stroke="var(--color-accent)" name="traces" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

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
        </>
      )}
    </div>
  );
}
