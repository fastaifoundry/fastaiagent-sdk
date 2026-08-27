import { Link } from "react-router-dom";
import { AlertTriangle, ArrowRight, CheckCircle2, Info, ShieldAlert } from "lucide-react";
import { cn } from "@/lib/utils";
import type { AttentionItem, Severity } from "./attention";

/**
 * "Needs attention" — the block Home leads with.
 *
 * States the problem in words, ranks by severity, and hands off to the page
 * that owns the fix. When there is nothing wrong it collapses to a single calm
 * line rather than a wall of green zeros: an all-clear should take one glance.
 *
 * Severity is never carried by colour alone — each card has an icon and a
 * written statement.
 */

const TONE: Record<Severity, { chip: string; icon: typeof AlertTriangle; ring: string }> = {
  critical: {
    chip: "bg-destructive/15 text-destructive",
    icon: ShieldAlert,
    ring: "border-destructive/30",
  },
  warning: {
    chip: "bg-fa-warning/15 text-fa-warning",
    icon: AlertTriangle,
    ring: "border-fa-warning/30",
  },
  info: {
    chip: "bg-secondary text-secondary-foreground",
    icon: Info,
    ring: "border-border",
  },
};

export function AttentionStrip({
  items,
  isLoading,
}: {
  items: AttentionItem[];
  isLoading?: boolean;
}) {
  if (isLoading) {
    return <div className="h-20 animate-pulse rounded-xl border border-border bg-card" />;
  }

  if (!items.length) {
    return (
      <div
        className="flex flex-wrap items-center gap-2.5 rounded-xl border border-border bg-card px-4 py-3"
        data-testid="attention-all-clear"
      >
        <CheckCircle2 className="h-4 w-4 shrink-0 text-fa-success" />
        <span className="text-sm font-medium">Nothing needs attention</span>
        <span className="text-[12.5px] text-muted-foreground">
          No failing traces, stalled executions or approvals waiting.
        </span>
      </div>
    );
  }

  return (
    <section className="space-y-2" data-testid="attention-strip">
      <div className="flex items-center gap-2">
        <h2 className="text-sm font-semibold">Needs attention</h2>
        <span className="rounded-full bg-secondary px-2 py-0.5 font-mono text-[10.5px] text-secondary-foreground">
          {items.length}
        </span>
      </div>
      <div className="grid gap-2.5 md:grid-cols-2 xl:grid-cols-3">
        {items.map((it) => {
          const tone = TONE[it.severity];
          const Icon = tone.icon;
          return (
            <Link
              key={it.key}
              to={it.to}
              className={cn(
                "group flex flex-col rounded-xl border bg-card p-3.5 text-left transition-colors hover:border-primary/50",
                tone.ring
              )}
            >
              <div className="flex items-start gap-2">
                <span className={cn("mt-0.5 rounded-md p-1", tone.chip)}>
                  <Icon className="h-3.5 w-3.5" />
                </span>
                <span className="min-w-0 flex-1 text-[13px] font-semibold leading-snug">
                  {it.title}
                </span>
              </div>
              <p className="mt-1.5 text-[12px] leading-snug text-muted-foreground">
                {it.detail}
              </p>
              <span className="mt-2 inline-flex items-center gap-1 text-[11.5px] font-medium text-primary">
                {it.cta}
                <ArrowRight className="h-3 w-3 transition-transform group-hover:translate-x-0.5" />
              </span>
            </Link>
          );
        })}
      </div>
    </section>
  );
}
