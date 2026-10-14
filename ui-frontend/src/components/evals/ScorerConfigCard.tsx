import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { EvalScorerConfig } from "@/lib/types";

/** The order a judge's settings read best in; anything else follows. */
const KEY_ORDER = [
  "criteria",
  "mode",
  "scale",
  "threshold",
  "provider",
  "model",
  "evaluation_steps",
  "steps",
  "rubric",
];

function valueText(v: unknown): string {
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  if (Array.isArray(v)) {
    // A rubric is a list of [score, description] pairs.
    return v
      .map((x) =>
        Array.isArray(x) && x.length === 2 ? `${x[0]}: ${x[1]}` : typeof x === "string" ? x : JSON.stringify(x)
      )
      .join("\n");
  }
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

/** How each scorer of a run was configured (`metadata.scorer_config`). */
export function ScorerConfigCard({ config }: { config: EvalScorerConfig | null | undefined }) {
  const entries = Object.entries(config ?? {}).filter(
    ([, c]) => c && typeof c === "object"
  );
  if (entries.length === 0) return null;
  return (
    <Card data-testid="scorer-config-card">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm">Scorer configuration</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-1 gap-3 text-xs md:grid-cols-2">
        {entries.map(([name, cfg]) => {
          const fields = Object.entries(cfg)
            .filter(([k, v]) => k !== "type" && v != null && v !== "")
            .sort(([a], [b]) => {
              const ia = KEY_ORDER.indexOf(a);
              const ib = KEY_ORDER.indexOf(b);
              return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
            });
          return (
            <div key={name} className="rounded-md border p-2.5">
              <div className="flex items-center justify-between gap-2">
                <span className="font-mono font-medium">{name}</span>
                {cfg.type && (
                  <span className="font-mono text-[10px] text-muted-foreground">{cfg.type}</span>
                )}
              </div>
              {fields.length > 0 ? (
                <dl className="mt-2 space-y-1">
                  {fields.map(([k, v]) => (
                    <div key={k} className="grid grid-cols-[110px_1fr] gap-2">
                      <dt className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                        {k.replace(/_/g, " ")}
                      </dt>
                      <dd className="whitespace-pre-wrap break-words font-mono">{valueText(v)}</dd>
                    </div>
                  ))}
                </dl>
              ) : (
                <p className="mt-1 text-muted-foreground">No settings.</p>
              )}
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}
