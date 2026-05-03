import { useState } from "react";
import { CalendarRange } from "lucide-react";
import { DayPicker, type DateRange } from "react-day-picker";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

import "react-day-picker/style.css";

interface Props {
  /** Current ``since`` ISO timestamp (filter state). */
  since?: string;
  /** Current ``until`` ISO timestamp (filter state). */
  until?: string;
  /** True when a custom range is active — controls trigger highlight. */
  active: boolean;
  onApply: (range: { since?: string; until?: string }) => void;
}

function toDate(iso?: string): Date | undefined {
  if (!iso) return undefined;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? undefined : d;
}

function startOfDay(d: Date): string {
  const x = new Date(d);
  x.setHours(0, 0, 0, 0);
  return x.toISOString();
}

function endOfDay(d: Date): string {
  const x = new Date(d);
  x.setHours(23, 59, 59, 999);
  return x.toISOString();
}

/**
 * Custom date-range picker for the trace filter bar. Opens a modal
 * with a two-month calendar; on apply, sets both ``since`` and
 * ``until`` to the day boundaries of the chosen range.
 *
 * Uses react-day-picker, dropped in once for Sprint 3 — same library
 * the shadcn Calendar component wraps, so the visual language stays
 * consistent if we adopt it elsewhere later.
 */
export function DateRangePopover({ since, until, active, onApply }: Props) {
  const [open, setOpen] = useState(false);
  const [range, setRange] = useState<DateRange | undefined>({
    from: toDate(since),
    to: toDate(until),
  });

  const handleApply = () => {
    onApply({
      since: range?.from ? startOfDay(range.from) : undefined,
      until: range?.to ? endOfDay(range.to) : undefined,
    });
    setOpen(false);
  };

  const handleClear = () => {
    setRange(undefined);
    onApply({ since: undefined, until: undefined });
    setOpen(false);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className={cn(
          "rounded px-2 py-1 text-xs font-mono font-medium transition-colors flex items-center gap-1",
          active
            ? "bg-primary text-primary-foreground"
            : "text-muted-foreground hover:text-foreground hover:bg-muted"
        )}
        title="Custom date range"
      >
        <CalendarRange className="h-3 w-3" />
        Custom
      </button>
      <DialogContent className="max-w-fit">
        <DialogHeader>
          <DialogTitle>Pick a date range</DialogTitle>
        </DialogHeader>
        <DayPicker
          mode="range"
          numberOfMonths={2}
          selected={range}
          onSelect={setRange}
          showOutsideDays
        />
        <DialogFooter className="gap-2 sm:justify-between">
          <Button variant="ghost" size="sm" onClick={handleClear}>
            Clear range
          </Button>
          <div className="flex gap-2">
            <Button variant="ghost" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleApply} disabled={!range?.from}>
              Apply
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
