/**
 * Reusable cost-breakdown table.
 *
 * Three modes — by model, by agent, by node — share the same shell and
 * differ only in column config. Sorted by cost (descending) on the
 * server, so we render rows in order.
 */
import { useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useCostBreakdown } from "@/hooks/use-cost-breakdown";
import type {
  CostByAgentRow,
  CostByModelRow,
  CostByNodeRow,
  CostGroupBy,
} from "@/lib/types";
import { formatCost, formatDurationMs } from "@/lib/format";

interface Props {
  groupBy: CostGroupBy;
  period: "1d" | "7d" | "30d" | "all";
  chainName?: string | null;
  /** Optional title override; defaults to a sensible per-group string. */
  title?: string;
}

const DEFAULT_TITLES: Record<CostGroupBy, string> = {
  model: "By model",
  agent: "By agent",
  node: "By node",
};

export function CostBreakdownTable({
  groupBy,
  period,
  chainName,
  title,
}: Props) {
  const { data, isLoading, error } = useCostBreakdown({
    groupBy,
    period,
    chainName,
  });
  const [sortDesc] = useState(true);

  return (
    <Card data-testid={`cost-breakdown-${groupBy}`}>
      <CardHeader>
        <CardTitle className="text-base">
          {title ?? DEFAULT_TITLES[groupBy]}
          {data?.period ? (
            <span className="ml-2 font-mono text-xs text-muted-foreground">
              · {data.period}
            </span>
          ) : null}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <p className="text-xs text-muted-foreground">Loading…</p>
        ) : error ? (
          <p className="text-xs text-destructive">
            {error instanceof Error ? error.message : String(error)}
          </p>
        ) : !data || data.rows.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            No spans in this window. Run an agent and refresh.
          </p>
        ) : groupBy === "model" ? (
          <ModelTable rows={data.rows as CostByModelRow[]} sortDesc={sortDesc} />
        ) : groupBy === "agent" ? (
          <AgentTable rows={data.rows as CostByAgentRow[]} sortDesc={sortDesc} />
        ) : (
          <NodeTable rows={data.rows as CostByNodeRow[]} sortDesc={sortDesc} />
        )}
      </CardContent>
    </Card>
  );
}

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function ModelTable({
  rows,
  sortDesc: _sortDesc,
}: {
  rows: CostByModelRow[];
  sortDesc: boolean;
}) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Model</TableHead>
          <TableHead className="text-right">Calls</TableHead>
          <TableHead className="text-right">Input tokens</TableHead>
          <TableHead className="text-right">Output tokens</TableHead>
          <TableHead className="text-right">Cost</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((r) => (
          <TableRow key={r.model}>
            <TableCell className="font-mono text-xs">{r.model}</TableCell>
            <TableCell className="text-right font-mono">{r.calls}</TableCell>
            <TableCell className="text-right font-mono">
              {fmtTokens(r.input_tokens)}
            </TableCell>
            <TableCell className="text-right font-mono">
              {fmtTokens(r.output_tokens)}
            </TableCell>
            <TableCell className="text-right font-mono">
              {formatCost(r.cost_usd)}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

function AgentTable({
  rows,
}: {
  rows: CostByAgentRow[];
  sortDesc: boolean;
}) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Agent</TableHead>
          <TableHead className="text-right">Runs</TableHead>
          <TableHead className="text-right">Avg tokens</TableHead>
          <TableHead className="text-right">Avg cost</TableHead>
          <TableHead className="text-right">Total cost</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((r) => (
          <TableRow key={r.agent}>
            <TableCell className="font-mono text-xs">{r.agent}</TableCell>
            <TableCell className="text-right font-mono">{r.runs}</TableCell>
            <TableCell className="text-right font-mono">
              {fmtTokens(r.avg_tokens)}
            </TableCell>
            <TableCell className="text-right font-mono">
              {formatCost(r.avg_cost_usd)}
            </TableCell>
            <TableCell className="text-right font-mono">
              {formatCost(r.total_cost_usd)}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

function NodeTable({
  rows,
}: {
  rows: CostByNodeRow[];
  sortDesc: boolean;
}) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Node</TableHead>
          <TableHead className="text-right">Executions</TableHead>
          <TableHead className="text-right">Avg duration</TableHead>
          <TableHead className="text-right">Avg cost</TableHead>
          <TableHead className="text-right">% of total</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((r) => (
          <TableRow key={r.node}>
            <TableCell className="font-mono text-xs">{r.node}</TableCell>
            <TableCell className="text-right font-mono">
              {r.executions}
            </TableCell>
            <TableCell className="text-right font-mono">
              {formatDurationMs(r.avg_duration_ms)}
            </TableCell>
            <TableCell className="text-right font-mono">
              {formatCost(r.avg_cost_usd)}
            </TableCell>
            <TableCell className="text-right font-mono">
              {r.percent_of_total.toFixed(1)}%
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
