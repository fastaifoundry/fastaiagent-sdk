/**
 * Plain key-level JSON diff for two adjacent checkpoint state snapshots.
 *
 * Returns three buckets — added, removed, changed — and renders them as
 * colored JSON blocks. We intentionally avoid a full structural diff
 * library: most checkpoint state is a flat ``dict[str, Any]`` and key-
 * level coloring is enough to spot what each node mutated.
 */

interface Props {
  prev: Record<string, unknown> | null;
  next: Record<string, unknown> | null;
}

function diff(
  prev: Record<string, unknown> | null,
  next: Record<string, unknown> | null
) {
  const a = prev ?? {};
  const b = next ?? {};
  const added: Record<string, unknown> = {};
  const removed: Record<string, unknown> = {};
  const changed: Record<string, { from: unknown; to: unknown }> = {};
  for (const k of Object.keys(b)) {
    if (!(k in a)) {
      added[k] = b[k];
    } else if (JSON.stringify(a[k]) !== JSON.stringify(b[k])) {
      changed[k] = { from: a[k], to: b[k] };
    }
  }
  for (const k of Object.keys(a)) {
    if (!(k in b)) removed[k] = a[k];
  }
  return { added, removed, changed };
}

function jsonBlock(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function StateDiff({ prev, next }: Props) {
  const { added, removed, changed } = diff(prev, next);
  const hasAny =
    Object.keys(added).length > 0 ||
    Object.keys(removed).length > 0 ||
    Object.keys(changed).length > 0;
  if (!hasAny) {
    return (
      <p className="text-xs text-muted-foreground">
        No state changes between these two checkpoints.
      </p>
    );
  }
  return (
    <div className="space-y-3 text-xs font-mono">
      {Object.keys(added).length > 0 ? (
        <Section
          label="Added"
          tone="bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border-emerald-500/30"
          body={jsonBlock(added)}
        />
      ) : null}
      {Object.keys(changed).length > 0 ? (
        <Section
          label="Changed"
          tone="bg-amber-500/10 text-amber-700 dark:text-amber-300 border-amber-500/30"
          body={jsonBlock(changed)}
        />
      ) : null}
      {Object.keys(removed).length > 0 ? (
        <Section
          label="Removed"
          tone="bg-red-500/10 text-red-700 dark:text-red-300 border-red-500/30"
          body={jsonBlock(removed)}
        />
      ) : null}
    </div>
  );
}

function Section({
  label,
  tone,
  body,
}: {
  label: string;
  tone: string;
  body: string;
}) {
  return (
    <div className={`rounded-md border px-3 py-2 ${tone}`}>
      <div className="mb-1 text-[10px] uppercase tracking-widest opacity-80">
        {label}
      </div>
      <pre className="whitespace-pre-wrap break-words">{body}</pre>
    </div>
  );
}
