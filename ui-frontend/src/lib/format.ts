/**
 * Small formatters shared across trace / eval / guardrail surfaces.
 * Keep logic here so display stays consistent.
 */

export function formatDurationMs(ms: number | null | undefined): string {
  if (ms == null) return "—";
  if (ms < 1) return "<1ms";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(s < 10 ? 2 : 1)}s`;
  const m = Math.floor(s / 60);
  const rem = Math.round(s - m * 60);
  return `${m}m ${rem}s`;
}

export function formatCost(usd: number | null | undefined): string {
  if (usd == null || usd === 0) return "—";
  if (usd < 0.001) return `$${(usd * 1000).toFixed(2)}m`; // millicents
  if (usd < 1) return `$${usd.toFixed(4)}`;
  return `$${usd.toFixed(2)}`;
}

export function formatTokens(n: number | null | undefined): string {
  if (n == null) return "—";
  if (n < 1000) return String(n);
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}k`;
  return `${(n / 1_000_000).toFixed(1)}M`;
}

export function formatTimeAgo(iso: string | null | undefined): string {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return iso;
  const diff = Date.now() - then;
  const sec = Math.max(0, Math.round(diff / 1000));
  if (sec < 10) return "just now";
  if (sec < 60) return `${sec}s ago`;
  const min = Math.round(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const days = Math.round(hr / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(iso).toLocaleDateString();
}

/**
 * Absolute local timestamp, compact enough for a table column.
 *
 * Deliberately separate from `formatTimeAgo` rather than replacing it: that
 * helper is shared by ~25 surfaces (evals, guardrails, KB, agents, playground…)
 * and changing it would silently restyle all of them.
 *
 * Seconds are included because traces frequently fire within the same minute —
 * minute precision would render consecutive runs as identical timestamps.
 *
 * The stored value carries a UTC offset (…+00:00), so this renders in the
 * viewer's local timezone, which is what you want when correlating a run
 * against something else that happened on this machine.
 */
export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

/** Full local timestamp for tooltips — includes the year and timezone name. */
export function formatDateTimeFull(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    dateStyle: "full",
    timeStyle: "long",
  });
}

export function shortTraceId(id: string): string {
  return id.length <= 10 ? id : `${id.slice(0, 8)}…${id.slice(-4)}`;
}

export function copyToClipboard(text: string): Promise<void> {
  if (typeof navigator === "undefined" || !navigator.clipboard) {
    return Promise.reject(new Error("Clipboard unavailable"));
  }
  return navigator.clipboard.writeText(text);
}
