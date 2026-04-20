import { useMemo } from "react";
import { Activity, Bot, ChevronRight, Database, Shield, Sparkles, Wrench } from "lucide-react";
import { cn } from "@/lib/utils";
import { formatDurationMs } from "@/lib/format";
import type { SpanRow, SpanTreeNode } from "@/lib/types";

interface Props {
  tree: SpanTreeNode;
  selectedSpanId: string | null;
  onSelect: (span: SpanRow) => void;
}

interface Layout {
  node: SpanTreeNode;
  depth: number;
  startMs: number;
  endMs: number;
}

function flatten(
  tree: SpanTreeNode,
  depth = 0,
  acc: Layout[] = [],
  origin: number | null = null
): { layout: Layout[]; origin: number; totalMs: number } {
  const startMs = new Date(tree.span.start_time).getTime();
  const endMs = new Date(tree.span.end_time).getTime();
  const safeStart = Number.isFinite(startMs) ? startMs : 0;
  const safeEnd = Number.isFinite(endMs) ? endMs : safeStart;
  const rootOrigin = origin ?? safeStart;
  acc.push({ node: tree, depth, startMs: safeStart, endMs: safeEnd });
  for (const child of tree.children) {
    flatten(child, depth + 1, acc, rootOrigin);
  }
  const totalMs = Math.max(...acc.map((l) => l.endMs)) - rootOrigin;
  return { layout: acc, origin: rootOrigin, totalMs: Math.max(totalMs, 1) };
}

function iconFor(name: string) {
  if (name.startsWith("agent.")) return Bot;
  if (name.startsWith("llm.") || name.startsWith("gen_ai.")) return Sparkles;
  if (name.startsWith("tool.")) return Wrench;
  if (name.startsWith("retrieval.") || name.startsWith("kb.")) return Database;
  if (name.startsWith("guardrail.")) return Shield;
  return Activity;
}

function barColor(name: string): string {
  if (name.startsWith("agent.")) return "bg-primary/70";
  if (name.startsWith("llm.") || name.startsWith("gen_ai.")) return "bg-accent/70";
  if (name.startsWith("tool.")) return "bg-fa-info/70";
  if (name.startsWith("retrieval.") || name.startsWith("kb.")) return "bg-fa-warning/70";
  if (name.startsWith("guardrail.")) return "bg-fa-success/70";
  return "bg-muted-foreground/40";
}

export function SpanTree({ tree, selectedSpanId, onSelect }: Props) {
  const { layout, origin, totalMs } = useMemo(() => flatten(tree), [tree]);

  return (
    <div className="rounded-md border bg-card">
      <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] border-b bg-muted/40 px-3 py-2 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        <span>Span</span>
        <span>Timeline</span>
      </div>
      <div className="divide-y">
        {layout.map(({ node, depth, startMs, endMs }) => {
          const span = node.span;
          const offsetPct = ((startMs - origin) / totalMs) * 100;
          const widthPct = Math.max(
            0.5,
            ((endMs - startMs) / totalMs) * 100
          );
          const Icon = iconFor(span.name);
          const active = span.span_id === selectedSpanId;
          const errored = (span.status || "OK").toUpperCase() === "ERROR";
          return (
            <button
              key={span.span_id}
              type="button"
              onClick={() => onSelect(span)}
              className={cn(
                "grid w-full grid-cols-[minmax(0,1fr)_minmax(0,1fr)] items-center gap-3 px-3 py-2 text-left text-sm transition-colors",
                active
                  ? "bg-primary/5"
                  : "hover:bg-muted/40"
              )}
            >
              <div
                className="flex items-center gap-1.5 min-w-0"
                style={{ paddingLeft: depth * 14 }}
              >
                {depth > 0 && (
                  <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground/60" />
                )}
                <Icon
                  className={cn(
                    "h-3.5 w-3.5 shrink-0",
                    errored ? "text-destructive" : "text-muted-foreground"
                  )}
                />
                <span className="truncate font-mono text-xs">{span.name}</span>
                {errored && (
                  <span className="ml-1 rounded bg-destructive/10 px-1 py-0.5 text-[10px] font-semibold uppercase text-destructive">
                    err
                  </span>
                )}
              </div>
              <div className="relative flex items-center gap-2 min-w-0">
                <div className="relative h-2 flex-1 rounded-full bg-muted">
                  <div
                    className={cn(
                      "absolute h-2 rounded-full",
                      barColor(span.name),
                      errored && "bg-destructive/60"
                    )}
                    style={{ left: `${offsetPct}%`, width: `${widthPct}%` }}
                  />
                </div>
                <span className="shrink-0 font-mono text-[11px] tabular-nums text-muted-foreground">
                  {formatDurationMs(endMs - startMs)}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
