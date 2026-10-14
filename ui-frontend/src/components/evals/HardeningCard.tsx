import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatDateTime } from "@/lib/format";
import type { EvalHardening } from "@/lib/types";

/** The fixes `harden()` recommended for this run's failures (`metadata.hardening`). */
export function HardeningCard({ hardening }: { hardening: EvalHardening | null | undefined }) {
  if (!hardening || typeof hardening !== "object") return null;
  const recs = Array.isArray(hardening.recommendations) ? hardening.recommendations : [];
  const meta = [
    hardening.failure_count != null
      ? `${hardening.failure_count} failure${hardening.failure_count === 1 ? "" : "s"} analysed`
      : null,
    hardening.model ? `by ${hardening.model}` : null,
    hardening.created_at ? formatDateTime(hardening.created_at) : null,
  ].filter(Boolean);
  return (
    <Card data-testid="hardening-card">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm">
          Hardening recommendations
          {meta.length > 0 && (
            <span className="ml-2 text-xs font-normal text-muted-foreground">
              {meta.join(" · ")}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-xs">
        {recs.length === 0 ? (
          <p className="text-muted-foreground">No recommendations were made.</p>
        ) : (
          recs.map((r, i) => (
            <div key={i} className="rounded-md border p-2.5">
              {r.target && (
                <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  {r.target}
                </div>
              )}
              {r.recommendation && (
                <p className="mt-1 whitespace-pre-wrap break-words">{r.recommendation}</p>
              )}
              {r.rationale && (
                <p className="mt-1 whitespace-pre-wrap break-words text-muted-foreground">
                  {r.rationale}
                </p>
              )}
            </div>
          ))
        )}
      </CardContent>
    </Card>
  );
}
