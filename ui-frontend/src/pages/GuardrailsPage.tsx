import { useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import {
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  EyeOff,
  Flag,
  RefreshCw,
  X,
} from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
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
import { DateRangePopover } from "@/components/traces/DateRangePopover";
import {
  GUARDRAIL_EVENTS_PAGE_SIZE,
  useGuardrailEvents,
} from "@/hooks/use-guardrails";
import { actionSummary, outcomeMeta, TONE_CHIP } from "@/lib/guardrail";
import { cn } from "@/lib/utils";
import { Timestamp } from "@/components/shared/Timestamp";

interface Filters {
  rule: string | null;
  outcome: string | null;
  agent: string | null;
  type: string | null;
  position: string | null;
  false_positive: boolean | null;
  since?: string;
  until?: string;
  /** On by default: a list of clean passes buries the checks that did something. */
  hide_passes: boolean;
  exclude_experiments: boolean;
}

const TYPE_OPTIONS = [
  "code",
  "regex",
  "llm_judge",
  "schema",
  "classifier",
  "content_safety",
  "groundedness",
  "topic",
  "pii",
  "secrets",
];

// Operator-assigned impact. Carried and shown; nothing branches on it.
const SEVERITY_CLASS: Record<string, string> = {
  low: "text-muted-foreground",
  medium: "text-fa-warning",
  high: "text-fa-warning",
  critical: "text-destructive",
};
const POSITION_OPTIONS = ["input", "output", "tool_call", "tool_result"];

const RANGES: { label: string; hours: number | null }[] = [
  { label: "1h", hours: 1 },
  { label: "24h", hours: 24 },
  { label: "7d", hours: 24 * 7 },
  { label: "30d", hours: 24 * 30 },
  { label: "All", hours: null },
];

function emptyFilters(): Filters {
  return {
    rule: null,
    outcome: null,
    agent: null,
    type: null,
    position: null,
    false_positive: null,
    hide_passes: true,
    exclude_experiments: false,
  };
}

export function GuardrailsPage() {
  // security_review_1.md L2 — SPA navigation via React Router instead
  // of ``window.location.assign`` (which forces a full reload AND would
  // become an open-redirect risk if ``row.event_id`` ever started
  // accepting external input).
  const navigate = useNavigate();
  // The agent page links here as ``/guardrails?agent=…&rule=…``.
  const [params] = useSearchParams();
  const [filters, setFilters] = useState<Filters>(() => ({
    ...emptyFilters(),
    rule: params.get("rule") || null,
    agent: params.get("agent") || null,
    outcome: params.get("outcome") || null,
    position: params.get("position") || null,
    type: params.get("type") || null,
  }));
  const [page, setPage] = useState(1);
  // Every filter change starts again from page 1.
  const update = (patch: Partial<Filters>) => {
    setFilters({ ...filters, ...patch });
    setPage(1);
  };

  const events = useGuardrailEvents({
    rule: filters.rule ?? undefined,
    outcome: filters.outcome ?? undefined,
    agent: filters.agent ?? undefined,
    type: filters.type ?? undefined,
    position: filters.position ?? undefined,
    false_positive: filters.false_positive,
    since: filters.since,
    until: filters.until,
    // Asking for passes while hiding them would always be empty.
    hide_passes: filters.hide_passes && filters.outcome !== "passed",
    exclude_experiments: filters.exclude_experiments,
    page,
    page_size: GUARDRAIL_EVENTS_PAGE_SIZE,
  });

  const rows = events.data?.rows ?? [];
  const total = events.data?.total ?? 0;
  const pageSize = events.data?.page_size ?? GUARDRAIL_EVENTS_PAGE_SIZE;
  const firstRow = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const lastRow = Math.min(page * pageSize, total);
  const legacySpans = events.data?.legacy_guardrail_spans ?? 0;

  const anyFilter =
    filters.rule ||
    filters.outcome ||
    filters.agent ||
    filters.type ||
    filters.position ||
    filters.false_positive != null ||
    filters.since ||
    filters.until ||
    filters.exclude_experiments;

  const isCustomRange = !!(filters.since && filters.until);
  const [quickRange, setQuickRange] = useState<string>("All");
  const setRange = (label: string, hours: number | null) => {
    setQuickRange(label);
    update({
      since: hours == null ? undefined : new Date(Date.now() - hours * 3600_000).toISOString(),
      until: undefined,
    });
  };

  return (
    <div className="space-y-5">
      <PageHeader
        title="Guardrail events"
        description={
          events.data
            ? `${total.toLocaleString()} event${total === 1 ? "" : "s"}${
                filters.hide_passes ? " (passes hidden)" : ""
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
          onChange={(e) => update({ rule: e.target.value || null })}
        />
        <select
          value={filters.outcome ?? ""}
          onChange={(e) => update({ outcome: e.target.value || null })}
          className="h-9 rounded-md border border-input bg-background px-2 text-sm"
        >
          <option value="">All outcomes</option>
          <option value="passed">Passed</option>
          <option value="blocked">Blocked</option>
          <option value="warned">Warned</option>
          <option value="filtered">Filtered</option>
          <option value="reasked">Re-asked</option>
          <option value="errored">Could not run (errored)</option>
        </select>
        <select
          value={filters.type ?? ""}
          onChange={(e) => update({ type: e.target.value || null })}
          className="h-9 rounded-md border border-input bg-background px-2 text-sm"
        >
          <option value="">All types</option>
          {TYPE_OPTIONS.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        <select
          value={filters.position ?? ""}
          onChange={(e) => update({ position: e.target.value || null })}
          className="h-9 rounded-md border border-input bg-background px-2 text-sm"
        >
          <option value="">All positions</option>
          {POSITION_OPTIONS.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
        <Input
          className="w-48"
          placeholder="Agent"
          value={filters.agent ?? ""}
          onChange={(e) => update({ agent: e.target.value || null })}
        />
        <select
          value={
            filters.false_positive == null
              ? ""
              : filters.false_positive
                ? "yes"
                : "no"
          }
          onChange={(e) => {
            const v = e.target.value;
            update({
              false_positive: v === "yes" ? true : v === "no" ? false : null,
            });
          }}
          className="h-9 rounded-md border border-input bg-background px-2 text-sm"
        >
          <option value="">FP: any</option>
          <option value="yes">FP: yes</option>
          <option value="no">FP: no</option>
        </select>

        <div className="flex items-center gap-1 rounded-md border bg-card p-0.5">
          {RANGES.map((r) => (
            <button
              key={r.label}
              type="button"
              onClick={() => setRange(r.label, r.hours)}
              className={cn(
                "rounded px-2 py-1 text-xs font-mono font-medium transition-colors",
                !isCustomRange && quickRange === r.label
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
              )}
            >
              {r.label}
            </button>
          ))}
          <DateRangePopover
            since={filters.since}
            until={filters.until}
            active={isCustomRange}
            onApply={({ since, until }) => {
              setQuickRange(since || until ? "Custom" : "All");
              update({ since, until });
            }}
          />
        </div>

        <label
          className="flex h-9 items-center gap-2 rounded-md border px-2.5 text-xs font-mono"
          title="Leave out clean passes so the checks that did something stand out."
        >
          <EyeOff className="h-3.5 w-3.5 text-muted-foreground" />
          <span>Hide passes</span>
          <Switch
            checked={filters.hide_passes}
            onCheckedChange={(v) => update({ hide_passes: v })}
            aria-label="Hide passes"
            size="sm"
          />
        </label>

        {/* Same chip as the trace list: hides checks from replay reruns,
            Playground calls and other non-production traces. */}
        <button
          type="button"
          aria-pressed={filters.exclude_experiments}
          onClick={() => update({ exclude_experiments: !filters.exclude_experiments })}
          title="Hide checks from replay, Playground, learn and eval traces"
          className={cn(
            "h-9 rounded-md border px-2.5 text-xs font-mono font-medium transition-colors",
            filters.exclude_experiments
              ? "border-primary bg-primary text-primary-foreground"
              : "bg-card text-muted-foreground hover:text-foreground hover:bg-muted"
          )}
        >
          Hide experiments
        </button>

        {anyFilter && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setQuickRange("All");
              setFilters({ ...emptyFilters(), hide_passes: filters.hide_passes });
              setPage(1);
            }}
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
          title={anyFilter ? "No guardrail checks match these filters" : "No guardrail checks recorded"}
          description={
            anyFilter
              ? `Try widening the time range or clearing filters.${
                  filters.hide_passes ? " Clean passes are hidden — turn off Hide passes to include them." : ""
                }`
              : "Checks are recorded whenever tracing is on " +
                "(FASTAIAGENT_TRACE_ENABLED, default on) and this UI reads the same local.db " +
                "as your agent (FASTAIAGENT_LOCAL_DB / --db)." +
                (filters.hide_passes
                  ? " Clean passes are hidden — turn off Hide passes to include them."
                  : "")
          }
          action={
            legacySpans > 0 ? (
              <p className="text-sm text-muted-foreground" data-testid="legacy-guardrail-spans">
                {legacySpans.toLocaleString()} guardrail span
                {legacySpans === 1 ? "" : "s"} exist in traces recorded before 1.89.0 —{" "}
                <Link to="/traces" className="text-primary hover:underline">
                  open Traces
                </Link>
                .
              </p>
            ) : undefined
          }
        />
      ) : (
        <div className="space-y-2">
          <div className="rounded-md border bg-card">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead>Rule</TableHead>
                  <TableHead className="w-[90px]">Type</TableHead>
                  <TableHead className="w-[100px]">Position</TableHead>
                  <TableHead className="w-[120px]">Outcome</TableHead>
                  <TableHead className="w-[150px]">Action</TableHead>
                  <TableHead className="w-[90px]">Severity</TableHead>
                  <TableHead className="w-[80px] text-right">Score</TableHead>
                  <TableHead>Agent</TableHead>
                  <TableHead>Message</TableHead>
                  <TableHead className="w-[120px]">When</TableHead>
                  <TableHead className="w-[60px]" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((row) => {
                  const meta = outcomeMeta(row.outcome);
                  const action = actionSummary(row);
                  return (
                    <TableRow
                      key={row.event_id}
                      className="cursor-pointer"
                      onClick={(e) => {
                        // Let inner links / icons handle their own clicks.
                        if ((e.target as HTMLElement).closest("a")) return;
                        navigate(`/guardrail-events/${row.event_id}`);
                      }}
                      data-testid={`guardrail-row-${row.event_id}`}
                    >
                      <TableCell className="font-medium font-mono text-xs">
                        <Link
                          to={`/guardrail-events/${row.event_id}`}
                          className="hover:text-primary"
                        >
                          {row.guardrail_name}
                        </Link>
                        {row.false_positive && (
                          <span className="ml-1 inline-flex items-center gap-0.5 rounded-sm bg-primary/10 px-1 py-px text-[9px] uppercase text-primary">
                            <Flag className="h-2.5 w-2.5" />
                            FP
                          </span>
                        )}
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
                            TONE_CHIP[meta.tone],
                          )}
                        >
                          <span className="h-1.5 w-1.5 rounded-full bg-current" />
                          {meta.label}
                        </span>
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        {action ? (
                          <span
                            className={cn("font-mono", action.diverged && "text-fa-warning")}
                            title={action.title}
                          >
                            {action.text}
                            {action.diverged && " *"}
                          </span>
                        ) : (
                          "—"
                        )}
                      </TableCell>
                      <TableCell className="text-xs">
                        <span
                          className={cn(
                            "font-mono",
                            SEVERITY_CLASS[row.severity ?? ""] ?? "text-muted-foreground",
                          )}
                        >
                          {row.severity ?? "—"}
                        </span>
                        {row.floor && (
                          <span
                            className="ml-1 rounded-sm bg-primary/10 px-1 py-px text-[9px] uppercase text-primary"
                            title="Organisation baseline — only a domain admin may change this rule"
                          >
                            floor
                          </span>
                        )}
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
                        <Timestamp iso={row.timestamp} />
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-1">
                          {row.trace_id && (
                            <Link
                              to={`/traces/${row.trace_id}`}
                              title="Open trace"
                              className="inline-flex items-center text-muted-foreground hover:text-primary"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <ExternalLink className="h-3.5 w-3.5" />
                            </Link>
                          )}
                          <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>

          <div className="flex items-center justify-end gap-2 text-xs text-muted-foreground">
            <span className="font-mono tabular-nums" data-testid="guardrail-pager-range">
              {firstRow.toLocaleString()}–{lastRow.toLocaleString()} of{" "}
              {total.toLocaleString()}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage(page - 1)}
              disabled={page <= 1 || events.isFetching}
            >
              <ChevronLeft className="mr-1 h-3.5 w-3.5" />
              Prev
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage(page + 1)}
              disabled={lastRow >= total || events.isFetching}
            >
              Next
              <ChevronRight className="ml-1 h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
