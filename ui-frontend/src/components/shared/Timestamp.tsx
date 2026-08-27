import { cn } from "@/lib/utils";
import { formatDateTime, formatDateTimeFull, formatTimeAgo } from "@/lib/format";

/**
 * An absolute, local, second-precision timestamp with the relative reading in
 * its tooltip.
 *
 * The rule this encodes: **"when did this happen" is absolute; "how stale is
 * this" is relative.**
 *
 * A "Started" column is read while correlating a run against a log line, a
 * deploy or another run — and "6h ago" can't be matched against any of them.
 * Worse, several consecutive runs all render as "6h ago", which destroys the
 * ordering the column exists to convey.
 *
 * By contrast, "last run 3d ago" or "updated 2w ago" answers a staleness
 * question, where the relative form is the more useful one. Those call sites
 * deliberately keep `formatTimeAgo`.
 */
export function Timestamp({
  iso,
  className,
}: {
  iso: string | null | undefined;
  className?: string;
}) {
  return (
    <span
      className={cn("font-mono tabular-nums", className)}
      title={
        iso ? `${formatDateTimeFull(iso)} · ${formatTimeAgo(iso)}` : undefined
      }
    >
      {formatDateTime(iso)}
    </span>
  );
}
