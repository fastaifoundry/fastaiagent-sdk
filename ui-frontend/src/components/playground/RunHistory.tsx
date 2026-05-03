import { Clock } from "lucide-react";
import { cn } from "@/lib/utils";
import { formatCost, formatDurationMs, formatTimeAgo } from "@/lib/format";

export interface PlaygroundHistoryEntry {
  id: string;
  timestamp: string;
  model: string;
  provider: string;
  response: string;
  latency_ms: number;
  cost_usd: number | null;
  tokens: { input: number; output: number };
  trace_id: string | null;
  // Captured config so re-loading restores the input panel exactly.
  prompt_template: string;
  system_prompt: string | null;
  variables: Record<string, string>;
}

interface Props {
  entries: PlaygroundHistoryEntry[];
  activeId: string | null;
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
        return (
          <li key={entry.id}>
            <button
              type="button"
              onClick={() => onSelect(entry)}
              className={cn(
                "flex w-full flex-col gap-1 px-3 py-2 text-left transition-colors",
                isActive ? "bg-primary/10 text-primary" : "hover:bg-muted/50",
              )}
            >
              <div className="flex items-center justify-between text-xs">
                <span className="font-mono">{entry.model}</span>
                <span className="flex items-center gap-1 text-muted-foreground">
                  <Clock className="h-3 w-3" />
                  {formatTimeAgo(entry.timestamp)}
                </span>
              </div>
              <div className="line-clamp-1 text-xs text-muted-foreground">
                {preview || "(empty)"}
              </div>
              <div className="flex gap-3 text-[10px] text-muted-foreground tabular-nums">
                <span>{formatDurationMs(entry.latency_ms)}</span>
                <span>{formatCost(entry.cost_usd)}</span>
                <span>
                  {entry.tokens.input}+{entry.tokens.output} tok
                </span>
              </div>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
