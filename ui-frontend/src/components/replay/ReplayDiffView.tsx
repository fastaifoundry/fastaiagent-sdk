import { AlertTriangle, CheckCircle2 } from "lucide-react";
import { JsonViewer } from "@/components/shared/JsonViewer";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { ComparisonResult } from "@/lib/types";

interface Props {
  comparison: ComparisonResult;
  originalOutput: unknown;
  newOutput: unknown;
}

function safeSpan(
  steps: ComparisonResult["original_steps"] | ComparisonResult["new_steps"],
  idx: number
) {
  return steps[idx] ?? null;
}

export function ReplayDiffView({ comparison, originalOutput, newOutput }: Props) {
  const maxSteps = Math.max(
    comparison.original_steps.length,
    comparison.new_steps.length
  );
  const diverged = comparison.diverged_at;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader className="flex flex-row items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
            <CardTitle className="text-sm">Original output</CardTitle>
          </CardHeader>
          <CardContent>
            <JsonViewer data={originalOutput ?? null} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-primary" />
            <CardTitle className="text-sm">Rerun output</CardTitle>
          </CardHeader>
          <CardContent>
            <JsonViewer data={newOutput ?? null} />
          </CardContent>
        </Card>
      </div>

      <div className="rounded-md border bg-card">
        <div className="flex items-center justify-between border-b px-4 py-2">
          <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
            Step-by-step comparison
          </span>
          {diverged != null && (
            <span className="rounded-md bg-primary/10 px-2 py-0.5 text-xs font-mono font-medium text-primary">
              diverged at step {diverged}
            </span>
          )}
        </div>
        <div className="divide-y">
          {Array.from({ length: maxSteps }, (_, idx) => {
            const a = safeSpan(comparison.original_steps, idx);
            const b = safeSpan(comparison.new_steps, idx);
            const divergedRow = diverged != null && idx >= diverged;
            const nameA = a?.span_name ?? "—";
            const nameB = b?.span_name ?? "—";
            const changed = nameA !== nameB;
            return (
              <div
                key={idx}
                className={cn(
                  "grid grid-cols-[48px_1fr_1fr] items-start gap-3 px-4 py-3",
                  divergedRow && "bg-primary/5"
                )}
              >
                <div className="font-mono text-xs text-muted-foreground">
                  {idx}
                </div>
                <StepColumn
                  label="Original"
                  name={nameA}
                  highlight={changed}
                />
                <StepColumn label="Rerun" name={nameB} highlight={changed} />
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function StepColumn({
  label,
  name,
  highlight,
}: {
  label: string;
  name: string;
  highlight: boolean;
}) {
  return (
    <div>
      <span className="mb-0.5 block text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        {label}
      </span>
      <span
        className={cn(
          "block truncate font-mono text-xs",
          highlight ? "text-primary font-semibold" : "text-foreground"
        )}
      >
        {name}
      </span>
    </div>
  );
}
