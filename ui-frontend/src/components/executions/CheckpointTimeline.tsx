/**
 * Vertical timeline of checkpoint rows for one durable execution.
 *
 * Each row collapses by default; expanding it reveals the
 * ``state_snapshot``, ``node_input``, and ``node_output`` JSON blobs.
 * If two adjacent rows are expanded simultaneously, a state diff card
 * lights up below the timeline showing what changed between them.
 *
 * Refresh-based — no SSE, no live polling. Click Refresh in the page
 * header to re-fetch.
 */
import { useState } from "react";
import {
  CheckCircle2,
  CircleDot,
  Loader2,
  PauseCircle,
  XCircle,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import type { ExecutionCheckpoint } from "@/hooks/use-execution";
import { JsonViewer } from "@/components/shared/JsonViewer";
import { StateDiff } from "./StateDiff";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface Props {
  checkpoints: ExecutionCheckpoint[];
}

const STATUS_LABEL: Record<string, string> = {
  completed: "completed",
  interrupted: "interrupted",
  failed: "failed",
  pending: "pending",
};

function StatusIcon({ status }: { status: string }) {
  if (status === "completed")
    return (
      <CheckCircle2
        className="h-4 w-4 text-emerald-600 dark:text-emerald-400"
        aria-label="completed"
      />
    );
  if (status === "interrupted")
    return (
      <PauseCircle
        className="h-4 w-4 text-amber-600 dark:text-amber-400"
        aria-label="interrupted"
      />
    );
  if (status === "failed")
    return (
      <XCircle
        className="h-4 w-4 text-red-600 dark:text-red-400"
        aria-label="failed"
      />
    );
  if (status === "pending")
    return (
      <Loader2
        className="h-4 w-4 text-muted-foreground animate-spin"
        aria-label="pending"
      />
    );
  return <CircleDot className="h-4 w-4 text-muted-foreground" />;
}

function StatusPill({ status }: { status: string }) {
  const tone =
    status === "completed"
      ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
      : status === "interrupted"
        ? "bg-amber-500/10 text-amber-700 dark:text-amber-300"
        : status === "failed"
          ? "bg-red-500/10 text-red-700 dark:text-red-300"
          : "bg-muted text-muted-foreground";
  return (
    <span className={`rounded px-2 py-0.5 text-[10px] font-medium ${tone}`}>
      {STATUS_LABEL[status] ?? status}
    </span>
  );
}

function formatTimestamp(iso: string | null): string {
  if (!iso) return "—";
  const dt = new Date(iso);
  if (Number.isNaN(dt.getTime())) return iso;
  return dt.toLocaleTimeString();
}

function ageText(iso: string | null): string | null {
  if (!iso) return null;
  const dt = new Date(iso).getTime();
  if (Number.isNaN(dt)) return null;
  const diff = Date.now() - dt;
  if (diff < 60_000) return `${Math.round(diff / 1000)}s ago`;
  if (diff < 3_600_000) return `${Math.round(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.round(diff / 3_600_000)}h ago`;
  return `${Math.round(diff / 86_400_000)}d ago`;
}

export function CheckpointTimeline({ checkpoints }: Props) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const toggle = (id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  // Find the first pair of adjacent expanded rows for the diff card.
  const expandedAdjacentPair = (() => {
    for (let i = 1; i < checkpoints.length; i++) {
      if (
        expanded.has(checkpoints[i - 1].checkpoint_id) &&
        expanded.has(checkpoints[i].checkpoint_id)
      ) {
        return [checkpoints[i - 1], checkpoints[i]] as const;
      }
    }
    return null;
  })();

  return (
    <div className="space-y-4" data-testid="checkpoint-timeline">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Checkpoints (chronological)</CardTitle>
        </CardHeader>
        <CardContent>
          <ol className="relative space-y-3">
            <span
              className="absolute left-2 top-3 bottom-3 w-px bg-border"
              aria-hidden="true"
            />
            {checkpoints.map((cp, idx) => {
              const isOpen = expanded.has(cp.checkpoint_id);
              return (
                <li
                  key={cp.checkpoint_id}
                  className="relative pl-8"
                  data-checkpoint-status={cp.status}
                >
                  <span className="absolute left-0 top-1.5 grid h-4 w-4 place-items-center rounded-full bg-card border">
                    <StatusIcon status={cp.status} />
                  </span>
                  <button
                    type="button"
                    className="flex w-full items-center gap-2 text-left"
                    onClick={() => toggle(cp.checkpoint_id)}
                    aria-expanded={isOpen}
                  >
                    <span className="font-mono text-xs text-muted-foreground">
                      Step {idx}
                    </span>
                    <span className="font-mono text-xs font-medium">
                      {cp.node_id}
                    </span>
                    <StatusPill status={cp.status} />
                    <span className="ml-auto flex items-center gap-2 text-[10px] font-mono text-muted-foreground">
                      <span>{formatTimestamp(cp.created_at)}</span>
                      {isOpen ? (
                        <ChevronDown className="h-3 w-3" />
                      ) : (
                        <ChevronRight className="h-3 w-3" />
                      )}
                    </span>
                  </button>
                  {cp.status === "interrupted" && cp.interrupt_reason ? (
                    <div className="mt-1 ml-0 text-[11px] text-amber-700 dark:text-amber-300">
                      Interrupt:{" "}
                      <span className="font-mono">{cp.interrupt_reason}</span>
                      {ageText(cp.created_at)
                        ? ` — waiting ${ageText(cp.created_at)}`
                        : ""}
                    </div>
                  ) : null}
                  {isOpen ? (
                    <div className="mt-3 space-y-3 rounded-md border bg-muted/20 p-3">
                      <Pane label="state_snapshot" data={cp.state_snapshot} />
                      {hasContent(cp.node_input) ? (
                        <Pane label="node_input" data={cp.node_input} />
                      ) : null}
                      {hasContent(cp.node_output) ? (
                        <Pane label="node_output" data={cp.node_output} />
                      ) : null}
                      {cp.interrupt_context && hasContent(cp.interrupt_context) ? (
                        <Pane
                          label="interrupt_context"
                          data={cp.interrupt_context}
                        />
                      ) : null}
                    </div>
                  ) : null}
                </li>
              );
            })}
          </ol>
        </CardContent>
      </Card>

      {expandedAdjacentPair ? (
        <Card data-testid="state-diff">
          <CardHeader>
            <CardTitle className="text-base">
              State diff —{" "}
              <span className="font-mono text-xs text-muted-foreground">
                {expandedAdjacentPair[0].node_id} → {expandedAdjacentPair[1].node_id}
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <StateDiff
              prev={expandedAdjacentPair[0].state_snapshot}
              next={expandedAdjacentPair[1].state_snapshot}
            />
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}

function hasContent(obj: unknown): boolean {
  if (!obj) return false;
  if (typeof obj !== "object") return Boolean(obj);
  return Object.keys(obj as object).length > 0;
}

function Pane({ label, data }: { label: string; data: unknown }) {
  return (
    <div>
      <div className="mb-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        {label}
      </div>
      <JsonViewer
        data={data as Record<string, unknown> | unknown[]}
        className="text-[11px]"
      />
    </div>
  );
}
