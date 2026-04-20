import { Search, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import type { RunnerType, TraceFilters as F } from "@/lib/types";

const RANGES: { label: string; hours: number | null }[] = [
  { label: "15m", hours: 0.25 },
  { label: "1h", hours: 1 },
  { label: "24h", hours: 24 },
  { label: "7d", hours: 24 * 7 },
  { label: "All", hours: null },
];

interface Props {
  filters: F;
  onChange: (next: F) => void;
}

export function TraceFiltersBar({ filters, onChange }: Props) {
  const setSince = (hours: number | null) => {
    if (hours == null) {
      const { since: _omitSince, ...rest } = filters;
      void _omitSince;
      onChange({ ...rest });
      return;
    }
    const since = new Date(Date.now() - hours * 3600_000).toISOString();
    onChange({ ...filters, since });
  };

  const currentRange =
    filters.since == null
      ? "All"
      : RANGES.find((r) => {
          if (r.hours == null) return false;
          const iso = new Date(Date.now() - r.hours * 3600_000).toISOString();
          return Math.abs(
            new Date(iso).getTime() - new Date(filters.since!).getTime()
          ) < 2_000;
        })?.label ?? "Custom";

  const anyActive =
    !!filters.agent ||
    !!filters.status ||
    !!filters.q ||
    !!filters.thread_id ||
    !!filters.runner_type ||
    !!filters.since;

  const RUNNERS: { value: RunnerType | null; label: string }[] = [
    { value: null, label: "All" },
    { value: "agent", label: "Agent" },
    { value: "chain", label: "Chain" },
    { value: "swarm", label: "Swarm" },
    { value: "supervisor", label: "Supervisor" },
  ];

  return (
    <div className="flex flex-wrap items-center gap-2">
      <div className="relative flex-1 min-w-[240px] max-w-md">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
        <Input
          value={filters.q ?? ""}
          onChange={(e) => onChange({ ...filters, q: e.target.value, page: 1 })}
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
        value={filters.thread_id ?? ""}
        onChange={(e) =>
          onChange({ ...filters, thread_id: e.target.value || null, page: 1 })
        }
        placeholder="Thread id"
        className="w-44"
      />

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
  );
}
