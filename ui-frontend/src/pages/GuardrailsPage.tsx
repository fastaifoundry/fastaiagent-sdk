import { useState } from "react";
import { Link } from "react-router-dom";
import { ExternalLink, RefreshCw, X } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
import { useGuardrailEvents } from "@/hooks/use-guardrails";
import { formatTimeAgo } from "@/lib/format";
import { cn } from "@/lib/utils";

interface Filters {
  rule: string | null;
  outcome: string | null;
  agent: string | null;
}

const OUTCOME_META: Record<string, { label: string; className: string }> = {
  passed: { label: "passed", className: "bg-fa-success/10 text-fa-success" },
  blocked: { label: "blocked", className: "bg-destructive/10 text-destructive" },
  warned: { label: "warned", className: "bg-fa-warning/10 text-fa-warning" },
};

export function GuardrailsPage() {
  const [filters, setFilters] = useState<Filters>({
    rule: null,
    outcome: null,
    agent: null,
  });
  const events = useGuardrailEvents({
    rule: filters.rule ?? undefined,
    outcome: filters.outcome ?? undefined,
    agent: filters.agent ?? undefined,
  });

  const rows = events.data?.rows ?? [];
  const anyFilter = filters.rule || filters.outcome || filters.agent;

  return (
    <div className="space-y-5">
      <PageHeader
        title="Guardrail events"
        description={
          events.data
            ? `${events.data.total.toLocaleString()} event${
                events.data.total === 1 ? "" : "s"
              }`
            : undefined
        }
      >
        <Button
          variant="outline"
          size="sm"
          onClick={() => events.refetch()}
          disabled={events.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${events.isFetching ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </PageHeader>

      <div className="flex flex-wrap items-center gap-2">
        <Input
          className="w-56"
          placeholder="Rule name"
          value={filters.rule ?? ""}
          onChange={(e) => setFilters({ ...filters, rule: e.target.value || null })}
        />
        <select
          value={filters.outcome ?? ""}
          onChange={(e) => setFilters({ ...filters, outcome: e.target.value || null })}
          className="h-9 rounded-md border border-input bg-background px-2 text-sm"
        >
          <option value="">All outcomes</option>
          <option value="passed">Passed</option>
          <option value="blocked">Blocked</option>
          <option value="warned">Warned</option>
        </select>
        <Input
          className="w-48"
          placeholder="Agent"
          value={filters.agent ?? ""}
          onChange={(e) => setFilters({ ...filters, agent: e.target.value || null })}
        />
        {anyFilter && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setFilters({ rule: null, outcome: null, agent: null })}
          >
            <X className="mr-1 h-3.5 w-3.5" />
            Clear
          </Button>
        )}
      </div>

      {events.isLoading ? (
        <TableSkeleton rows={8} />
      ) : rows.length === 0 ? (
        <EmptyState
          title="No guardrail events"
          description="Events are logged when agents run with guardrails and UI is enabled."
        />
      ) : (
        <div className="rounded-md border bg-card">
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead>Rule</TableHead>
                <TableHead className="w-[90px]">Type</TableHead>
                <TableHead className="w-[100px]">Position</TableHead>
                <TableHead className="w-[110px]">Outcome</TableHead>
                <TableHead className="w-[80px] text-right">Score</TableHead>
                <TableHead>Agent</TableHead>
                <TableHead>Message</TableHead>
                <TableHead className="w-[120px]">When</TableHead>
                <TableHead className="w-[60px]" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => {
                const meta =
                  OUTCOME_META[(row.outcome ?? "").toLowerCase()] ?? {
                    label: row.outcome ?? "—",
                    className: "bg-muted text-muted-foreground",
                  };
                return (
                  <TableRow key={row.event_id}>
                    <TableCell className="font-medium font-mono text-xs">
                      {row.guardrail_name}
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {row.guardrail_type ?? "—"}
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {row.position ?? "—"}
                    </TableCell>
                    <TableCell>
                      <span
                        className={cn(
                          "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs font-mono uppercase",
                          meta.className
                        )}
                      >
                        <span className="h-1.5 w-1.5 rounded-full bg-current" />
                        {meta.label}
                      </span>
                    </TableCell>
                    <TableCell className="text-right font-mono tabular-nums text-xs">
                      {row.score != null ? row.score.toFixed(2) : "—"}
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {row.agent_name ?? "—"}
                    </TableCell>
                    <TableCell
                      className="max-w-[240px] truncate text-xs"
                      title={row.message ?? ""}
                    >
                      {row.message ?? "—"}
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {formatTimeAgo(row.timestamp)}
                    </TableCell>
                    <TableCell>
                      {row.trace_id && (
                        <Link
                          to={`/traces/${row.trace_id}`}
                          title="Open trace"
                          className="inline-flex items-center text-muted-foreground hover:text-primary"
                        >
                          <ExternalLink className="h-3.5 w-3.5" />
                        </Link>
                      )}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}
