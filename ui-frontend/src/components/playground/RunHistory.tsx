import { Clock } from "lucide-react";
import { cn } from "@/lib/utils";
import { formatCost, formatDurationMs, formatTimeAgo } from "@/lib/format";
import type { PlaygroundHistoryEntry } from "@/lib/playground-run";

interface Props {
  entries: PlaygroundHistoryEntry[];
  activeId: string | null;
  /** Restores the run's form and shows its response (1.89.0, P10d). */
  onSelect: (entry: PlaygroundHistoryEntry) => void;
}

export function RunHistory({ entries, activeId, onSelect }: Props) {
  if (entries.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        Runs you do in this session will appear here. History clears on
        refresh — playground is for quick experiments.
      </p>
    );
  }
  return (
    <ul className="divide-y rounded-md border bg-card">
      {entries.map((entry) => {
        const preview = entry.response.slice(0, 50);
        const isActive = activeId === entry.id;
        const { metadata } = entry;
        return (
          <li key={entry.id}>
            <button
              type="button"
              onClick={() => onSelect(entry)}
              title="Restore this run's prompt, model and parameters"
              className={cn(
                "flex w-full flex-col gap-1 px-3 py-2 text-left transition-colors",
                isActive ? "bg-primary/10 text-primary" : "hover:bg-muted/50",
              )}
            >
              <div className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-1.5">
                  <span className="font-mono">{metadata.model}</span>
                  {metadata.truncated && (
                    <span
                      className="rounded bg-amber-500/15 px-1 text-[10px] text-amber-700 dark:text-amber-300"
                      data-testid="history-truncated"
                    >
                      truncated
                    </span>
                  )}
                  {entry.request.prompt_ref && (
                    <span className="rounded border px-1 font-mono text-[10px] text-muted-foreground">
                      {entry.request.prompt_ref.slug}
                    </span>
                  )}
                </span>
                <span className="flex items-center gap-1 text-muted-foreground">
                  <Clock className="h-3 w-3" />
                  {formatTimeAgo(entry.timestamp)}
                </span>
              </div>
              <div className="line-clamp-1 text-xs text-muted-foreground">
                {preview || "(empty)"}
              </div>
              <div className="flex gap-3 text-[10px] text-muted-foreground tabular-nums">
                <span>{formatDurationMs(metadata.latency_ms)}</span>
                <span>{formatCost(metadata.cost_usd)}</span>
                <span>
                  {metadata.tokens.input}+{metadata.tokens.output} tok
                </span>
              </div>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
