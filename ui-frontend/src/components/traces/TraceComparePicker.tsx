import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { GitCompareArrows, Search } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useTraces } from "@/hooks/use-traces";
import {
  formatCost,
  formatDurationMs,
  formatTimeAgo,
  shortTraceId,
} from "@/lib/format";
import type { TraceRow } from "@/lib/types";

interface Props {
  /** The trace already chosen (the "left" / A side). Excluded from the picker. */
  pinnedTraceId: string;
  /** Optional override of the trigger button label. */
  triggerLabel?: string;
}

/**
 * Modal that drives "Compare with…" from the trace detail page. Lists the
 * 50 most-recent traces in the project (excluding the pinned one) so the
 * user can pick the second side of the comparison in one click.
 */
export function TraceComparePicker({ pinnedTraceId, triggerLabel }: Props) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const navigate = useNavigate();
  const { data, isLoading } = useTraces({ q, page: 1, page_size: 50 });

  const candidates = (data?.rows ?? []).filter(
    (r) => r.trace_id !== pinnedTraceId
  );

  const choose = (other: TraceRow) => {
    setOpen(false);
    navigate(
      `/traces/compare?a=${encodeURIComponent(pinnedTraceId)}&b=${encodeURIComponent(other.trace_id)}`
    );
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <GitCompareArrows className="mr-1.5 h-3.5 w-3.5" />
          {triggerLabel ?? "Compare with…"}
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Pick a trace to compare against</DialogTitle>
          <DialogDescription>
            The current trace stays on the left. The trace you choose here
            appears on the right of the comparison view.
          </DialogDescription>
        </DialogHeader>
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
          <Input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search recent traces…"
            className="pl-8"
            autoFocus
          />
        </div>
        <div className="max-h-[420px] overflow-y-auto rounded-md border">
          {isLoading ? (
            <p className="p-4 text-sm text-muted-foreground">Loading traces…</p>
          ) : candidates.length === 0 ? (
            <p className="p-4 text-sm text-muted-foreground">
              No other traces match.
            </p>
          ) : (
            <ul className="divide-y">
              {candidates.map((row) => (
                <li key={row.trace_id}>
                  <button
                    type="button"
                    onClick={() => choose(row)}
                    className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left hover:bg-muted/50"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="truncate font-medium text-sm">
                        {row.name || row.trace_id}
                      </div>
                      <div className="text-[11px] font-mono text-muted-foreground">
                        {shortTraceId(row.trace_id)} ·{" "}
                        {formatTimeAgo(row.start_time)}
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-3 text-[11px] font-mono tabular-nums text-muted-foreground">
                      <span>{formatDurationMs(row.duration_ms)}</span>
                      <span>{formatCost(row.total_cost_usd)}</span>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={() => setOpen(false)}>
            Cancel
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
