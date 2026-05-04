import { useEffect, useState } from "react";
import { Search, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { DateRangePopover } from "./DateRangePopover";
import { FilterPresetsMenu } from "./FilterPresetsMenu";
import { MoreFiltersPanel } from "./MoreFiltersPanel";
import type { RunnerType, TraceFilters as F } from "@/lib/types";

const RANGES: { label: string; hours: number | null }[] = [
  { label: "15m", hours: 0.25 },
  { label: "1h", hours: 1 },
  { label: "24h", hours: 24 },
  { label: "7d", hours: 24 * 7 },
  { label: "30d", hours: 24 * 30 },
  { label: "All", hours: null },
];

interface Props {
  filters: F;
  onChange: (next: F) => void;
}

/**
 * Trace filter bar. Sprint 3 enhancements:
 *
 *   - 30d quick range + Custom date picker (`DateRangePopover`).
 *   - Saved presets dropdown + Save/Manage (``FilterPresetsMenu``).
 *   - Collapsible duration/cost ranges (``MoreFiltersPanel``).
 *   - Debounced full-text search — 300ms after typing stops, the
 *     wider filter object updates (which the page lifts into the URL).
 */
export function TraceFiltersBar({ filters, onChange }: Props) {
  // Local mirror of the search box so we can debounce. Stay in sync
  // when filters.q changes from outside (preset load, URL navigation).
  const [searchText, setSearchText] = useState(filters.q ?? "");
  useEffect(() => {
    setSearchText(filters.q ?? "");
  }, [filters.q]);

  useEffect(() => {
    if ((filters.q ?? "") === searchText) return;
    const id = setTimeout(() => {
      onChange({ ...filters, q: searchText, page: 1 });
    }, 300);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchText]);

  const setSince = (hours: number | null) => {
    if (hours == null) {
      const { since: _omitSince, until: _omitUntil, ...rest } = filters;
      void _omitSince;
      void _omitUntil;
      onChange({ ...rest, page: 1 });
      return;
    }
    const since = new Date(Date.now() - hours * 3600_000).toISOString();
    // Quick ranges always clear ``until`` so the meaning is "the last N
    // units up to now".
    const { until: _u, ...rest } = filters;
    void _u;
    onChange({ ...rest, since, page: 1 });
  };

  const isCustomRange = !!(filters.since && filters.until);

  const currentRange = isCustomRange
    ? "Custom"
    : filters.since == null
      ? "All"
      : (RANGES.find((r) => {
          if (r.hours == null) return false;
          const iso = new Date(Date.now() - r.hours * 3600_000).toISOString();
          return (
            Math.abs(
              new Date(iso).getTime() - new Date(filters.since!).getTime()
            ) < 2_000
          );
        })?.label ?? "Custom");

  const anyActive =
    !!filters.agent ||
    !!filters.status ||
    !!filters.q ||
    !!filters.thread_id ||
    !!filters.runner_type ||
    !!filters.framework ||
    !!filters.since ||
    !!filters.until ||
    filters.min_duration_ms != null ||
    filters.max_duration_ms != null ||
    filters.min_cost != null ||
    filters.max_cost != null;

  const RUNNERS: { value: RunnerType | null; label: string }[] = [
    { value: null, label: "All" },
    { value: "agent", label: "Agent" },
    { value: "chain", label: "Chain" },
    { value: "swarm", label: "Swarm" },
    { value: "supervisor", label: "Supervisor" },
  ];

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative flex-1 min-w-[240px] max-w-md">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
          <Input
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            placeholder="Search trace name, input, output…"
            className="pl-8"
          />
        </div>

        <div className="flex items-center gap-1 rounded-md border bg-card p-0.5">
          {RANGES.map((r) => (
            <button
              key={r.label}
              type="button"
              onClick={() => setSince(r.hours)}
              className={cn(
                "rounded px-2 py-1 text-xs font-mono font-medium transition-colors",
                currentRange === r.label
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
            onApply={({ since, until }) =>
              onChange({ ...filters, since, until, page: 1 })
            }
          />
        </div>

        <select
          value={filters.status ?? ""}
          onChange={(e) =>
            onChange({ ...filters, status: e.target.value || null, page: 1 })
          }
          className="h-9 rounded-md border border-input bg-background px-2 text-sm"
        >
          <option value="">All statuses</option>
          <option value="OK">OK</option>
          <option value="ERROR">Error</option>
        </select>

        <Input
          value={filters.agent ?? ""}
          onChange={(e) =>
            onChange({ ...filters, agent: e.target.value || null, page: 1 })
          }
          placeholder="Agent name"
          className="w-44"
        />

        <div className="flex items-center gap-1 rounded-md border bg-card p-0.5">
          {RUNNERS.map((r) => (
            <button
              key={r.label}
              type="button"
              onClick={() =>
                onChange({ ...filters, runner_type: r.value, page: 1 })
              }
              className={cn(
                "rounded px-2 py-1 text-[11px] font-mono font-medium uppercase tracking-wider transition-colors",
                (filters.runner_type ?? null) === r.value
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
              )}
            >
              {r.label}
            </button>
          ))}
        </div>

        <Input
          value={filters.framework ?? ""}
          onChange={(e) =>
            onChange({ ...filters, framework: e.target.value || null, page: 1 })
          }
          placeholder="Framework"
          className="w-44"
          // Free-text on purpose: new frameworks (LangSmith, AutoGen,
          // future integrations) work without UI changes — just type
          // whatever value lands in ``fastaiagent.framework`` on the
          // root span.
        />


        <Input
          value={filters.thread_id ?? ""}
          onChange={(e) =>
            onChange({ ...filters, thread_id: e.target.value || null, page: 1 })
          }
          placeholder="Thread id"
          className="w-44"
        />

        <FilterPresetsMenu filters={filters} onApply={onChange} />

        {anyActive && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() =>
              onChange({ page: 1, page_size: filters.page_size ?? 100 })
            }
          >
            <X className="mr-1 h-3.5 w-3.5" /> Clear
          </Button>
        )}
      </div>

      <MoreFiltersPanel filters={filters} onChange={onChange} />
    </div>
  );
}
