import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { TraceFilters } from "@/lib/types";

interface Props {
  filters: TraceFilters;
  onChange: (next: TraceFilters) => void;
}

function parseNumber(value: string): number | undefined {
  if (value === "") return undefined;
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
}

/**
 * Collapsible "More filters" disclosure with duration + cost ranges.
 * Defaults closed; the toggle shows a count of active filters so users
 * notice when they have something hidden behind the disclosure.
 */
export function MoreFiltersPanel({ filters, onChange }: Props) {
  const activeCount =
    (filters.min_duration_ms != null ? 1 : 0) +
    (filters.max_duration_ms != null ? 1 : 0) +
    (filters.min_cost != null ? 1 : 0) +
    (filters.max_cost != null ? 1 : 0);
  const [open, setOpen] = useState(activeCount > 0);

  return (
    <div className="rounded-md border bg-card">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-xs font-mono uppercase tracking-widest text-muted-foreground hover:bg-muted/50"
      >
        {open ? (
          <ChevronDown className="h-3 w-3" />
        ) : (
          <ChevronRight className="h-3 w-3" />
        )}
        More filters
        {activeCount > 0 && (
          <span className="ml-2 rounded-md bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium text-primary">
            {activeCount} active
          </span>
        )}
      </button>
      {open && (
        <div className="grid gap-3 border-t px-3 py-3 sm:grid-cols-2">
          <div className="space-y-1">
            <Label className="text-[10px] uppercase tracking-widest text-muted-foreground">
              Duration (ms)
            </Label>
            <div className="flex items-center gap-2">
              <Input
                type="number"
                min={0}
                placeholder="min"
                value={filters.min_duration_ms ?? ""}
                onChange={(e) =>
                  onChange({
                    ...filters,
                    min_duration_ms: parseNumber(e.target.value),
                    page: 1,
                  })
                }
              />
              <span className="text-xs text-muted-foreground">—</span>
              <Input
                type="number"
                min={0}
                placeholder="max"
                value={filters.max_duration_ms ?? ""}
                onChange={(e) =>
                  onChange({
                    ...filters,
                    max_duration_ms: parseNumber(e.target.value),
                    page: 1,
                  })
                }
              />
            </div>
          </div>
          <div className="space-y-1">
            <Label className="text-[10px] uppercase tracking-widest text-muted-foreground">
              Cost (USD)
            </Label>
            <div className="flex items-center gap-2">
              <Input
                type="number"
                min={0}
                step={0.01}
                placeholder="min"
                value={filters.min_cost ?? ""}
                onChange={(e) =>
                  onChange({
                    ...filters,
                    min_cost: parseNumber(e.target.value),
                    page: 1,
                  })
                }
              />
              <span className="text-xs text-muted-foreground">—</span>
              <Input
                type="number"
                min={0}
                step={0.01}
                placeholder="max"
                value={filters.max_cost ?? ""}
                onChange={(e) =>
                  onChange({
                    ...filters,
                    max_cost: parseNumber(e.target.value),
                    page: 1,
                  })
                }
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
