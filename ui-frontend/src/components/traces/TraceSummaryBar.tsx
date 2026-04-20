import { Copy } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { TraceStatusBadge } from "./TraceStatusBadge";
import { copyToClipboard, formatCost, formatDurationMs, formatTimeAgo, formatTokens } from "@/lib/format";
import type { TraceDetail } from "@/lib/types";

interface Props {
  trace: TraceDetail;
  duration_ms: number | null;
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        {label}
      </span>
      <span className="text-sm font-mono tabular-nums">{value}</span>
    </div>
  );
}

export function TraceSummaryBar({ trace, duration_ms }: Props) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-4 rounded-md border bg-card px-4 py-3">
      <div className="flex flex-wrap items-center gap-6">
        <div className="flex flex-col">
          <div className="flex items-center gap-2">
            <span className="font-mono text-sm">{trace.trace_id}</span>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={async () => {
                try {
                  await copyToClipboard(trace.trace_id);
                  toast.success("Trace ID copied");
                } catch {
                  toast.error("Copy failed");
                }
              }}
            >
              <Copy className="h-3 w-3" />
            </Button>
          </div>
          <span className="text-xs text-muted-foreground">
            started {formatTimeAgo(trace.start_time)}
          </span>
        </div>
        <TraceStatusBadge status={trace.status} />
      </div>
      <div className="flex flex-wrap items-center gap-8">
        <Stat label="Agent" value={trace.agent_name ?? "—"} />
        <Stat label="Duration" value={formatDurationMs(duration_ms)} />
        <Stat label="Spans" value={String(trace.span_count)} />
        <Stat label="Tokens" value={formatTokens(trace.total_tokens)} />
        <Stat label="Cost" value={formatCost(trace.total_cost_usd)} />
      </div>
    </div>
  );
}
