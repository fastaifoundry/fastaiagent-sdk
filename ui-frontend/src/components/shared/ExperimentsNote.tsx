/**
 * "Include experiments" toggle and the note that says what was left out.
 *
 * Production views (Overview, Analytics, Agents) exclude traces whose
 * ``fastaiagent.source`` is a replay, Playground call, learn pass, eval run
 * and so on (1.89.0). Excluding them must never hide real spend, so whenever
 * any were left out the page says how many and what they cost.
 */
import { FlaskConical } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { formatCost } from "@/lib/format";
import { cn } from "@/lib/utils";
import { usePreferencesStore } from "@/stores/preferences-store";

export function IncludeExperimentsToggle({ className }: { className?: string }) {
  const includeExperiments = usePreferencesStore((s) => s.includeExperiments);
  const setIncludeExperiments = usePreferencesStore((s) => s.setIncludeExperiments);
  return (
    <label
      className={cn(
        "flex items-center gap-2 rounded-md border px-2.5 py-1 text-xs font-mono",
        className
      )}
      title={
        "Count replay, Playground, learn and eval traces as production. " +
        "Off by default: they are reported separately."
      }
    >
      <FlaskConical className="h-3.5 w-3.5 text-muted-foreground" />
      <span>Include experiments</span>
      <Switch
        checked={includeExperiments}
        onCheckedChange={setIncludeExperiments}
        aria-label="Include experiments"
        size="sm"
      />
    </label>
  );
}

/**
 * For views whose endpoint reports no experiments summary (the Agents pages):
 * say the numbers are production-only while experiments are excluded, and
 * nothing when they are included.
 */
export function ProductionOnlyNote({ className }: { className?: string }) {
  const includeExperiments = usePreferencesStore((s) => s.includeExperiments);
  if (includeExperiments) return null;
  return (
    <p
      data-testid="production-only-note"
      className={cn("text-xs text-muted-foreground", className)}
    >
      Counts production runs only; replay reruns and Playground calls are
      excluded.
    </p>
  );
}

interface NoteProps {
  /** Experiment traces in the window. Nothing renders when 0. */
  count: number | null | undefined;
  /** Whether the numbers on the page already include them. */
  included: boolean;
  /** Their cost, when the endpoint reports it. */
  costUsd?: number | null;
  bySource?: Record<string, { trace_count: number }> | null;
  /** Appended to the trace count, e.g. "from the last 24h". */
  suffix?: string;
  className?: string;
}

export function ExperimentsNote({
  count,
  included,
  costUsd,
  bySource,
  suffix,
  className,
}: NoteProps) {
  if (!count) return null;

  const sources = Object.entries(bySource ?? {})
    .filter(([, v]) => v.trace_count > 0)
    .sort((a, b) => b[1].trace_count - a[1].trace_count)
    .map(([source, v]) => `${source} ${v.trace_count}`)
    .join(", ");
  const details = [
    costUsd != null ? (costUsd > 0 ? formatCost(costUsd) : "$0.00") : null,
    sources || null,
  ]
    .filter(Boolean)
    .join(": ");

  return (
    <p
      data-testid="experiments-note"
      className={cn("text-xs text-muted-foreground", className)}
    >
      {included ? "Includes" : "Excludes"} {count.toLocaleString()} experiment
      trace{count === 1 ? "" : "s"}
      {suffix ? ` ${suffix}` : ""}
      {details ? ` (${details})` : ""}
      {included ? "." : ". Turn on Include experiments to count them."}
    </p>
  );
}
