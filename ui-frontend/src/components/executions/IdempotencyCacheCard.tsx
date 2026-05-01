/**
 * Read-only listing of cached ``@idempotent`` results for an execution.
 *
 * Each row is one ``(function_key, result)`` pair the SDK persisted on a
 * prior run. Resuming the execution would skip these calls. Empty
 * caches collapse to a one-line note so the page stays compact.
 */
import { useIdempotencyCache } from "@/hooks/use-idempotency-cache";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatTimeAgo } from "@/lib/format";

interface Props {
  executionId: string;
}

export function IdempotencyCacheCard({ executionId }: Props) {
  const { data, isLoading, error } = useIdempotencyCache(executionId);

  return (
    <Card data-testid="idempotency-cache">
      <CardHeader>
        <CardTitle className="text-base">
          Idempotency cache
          {data ? (
            <span className="ml-2 font-mono text-xs text-muted-foreground">
              ({data.count})
            </span>
          ) : null}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <p className="text-xs text-muted-foreground">Loading…</p>
        ) : error ? (
          <p className="text-xs text-destructive">
            {error instanceof Error ? error.message : String(error)}
          </p>
        ) : !data || data.count === 0 ? (
          <p className="text-xs text-muted-foreground">
            No cached <code className="font-mono">@idempotent</code> results
            for this execution. Decorate side-effecting helpers with{" "}
            <code className="font-mono">@idempotent</code> so resumes skip
            already-completed work.
          </p>
        ) : (
          <ul className="space-y-2">
            {data.items.map((row) => (
              <li
                key={row.function_key}
                className="rounded-md border bg-muted/30 px-3 py-2"
              >
                <div className="font-mono text-xs font-medium">
                  {row.function_key}
                </div>
                <div className="mt-1 flex items-center gap-2 text-[11px] text-muted-foreground">
                  <span className="font-mono">→</span>
                  <code className="font-mono truncate">
                    {jsonInline(row.result)}
                  </code>
                  <span>·</span>
                  <span>cached {formatTimeAgo(row.created_at)}</span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function jsonInline(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}
