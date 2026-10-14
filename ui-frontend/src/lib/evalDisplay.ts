/**
 * Display helpers for eval numbers (1.89.0).
 *
 * Two rates, cases first: the **cases** rate (passed cases over passed +
 * failed cases) is the headline; the **checks** rate (`run.pass_rate`) is what
 * `overall.pass_rate` and `--eval-fail-under` gate. A `null` rate means no
 * verdict applied — it renders "—", never 0%.
 */
import {
  evalVerdictOutcome,
  type EvalBytesDescriptor,
  type EvalMediaDescriptor,
  type EvalRunRow,
  type EvalTrendPoint,
  type EvalVerdict,
} from "@/lib/types";
import { formatCost } from "@/lib/format";

function finite(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function fmtNum(n: number): string {
  return Number.isInteger(n) ? String(n) : n.toFixed(2);
}

/**
 * A verdict's score as text. `null` for errored / n/a: their 0 is a
 * placeholder, never a score. For a judge with recorded details:
 * "3/5 → 0.50 (≥0.50)".
 */
export function verdictScoreText(v: EvalVerdict): string | null {
  const outcome = evalVerdictOutcome(v);
  if (outcome === "errored" || outcome === "skipped") return null;
  const score = finite(v.score);
  if (score == null) return null;
  const d = v.details ?? {};
  const raw = finite(d.raw_score);
  const threshold = finite(d.threshold);
  const scale = typeof d.scale === "string" ? d.scale : null;
  const bound = threshold != null ? ` (≥${threshold.toFixed(2)})` : "";
  if (raw != null && scale) {
    const m = /^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$/.exec(scale);
    // "0-1" and "binary" are already on the unit scale: no "raw/hi" prefix.
    if (m && Number(m[2]) !== 1) {
      return `${fmtNum(raw)}/${fmtNum(Number(m[2]))} → ${score.toFixed(2)}${bound}`;
    }
  }
  return `${score.toFixed(2)}${bound}`;
}

/** `0.8333` → `"83%"`; `null` → `"—"`. */
export function formatRate(rate: number | null | undefined): string {
  return typeof rate === "number" && Number.isFinite(rate) ? `${Math.round(rate * 100)}%` : "—";
}

/** A signed percentage-point delta: `+5%`, `-12%`, `"—"` for null. */
export function formatRateDelta(delta: number | null | undefined): string {
  if (typeof delta !== "number" || !Number.isFinite(delta)) return "—";
  const pct = Math.round(delta * 100);
  return `${pct > 0 ? "+" : ""}${pct}%`;
}

/** A signed cost delta; `"—"` when either side was unknown. */
export function formatCostDelta(delta: number | null | undefined): string {
  if (typeof delta !== "number" || !Number.isFinite(delta)) return "—";
  if (delta === 0) return "$0";
  const sign = delta > 0 ? "+" : "-";
  return `${sign}${formatCost(Math.abs(delta))}`;
}

/** Case counts of a run, falling back to 0 for legacy rows. */
export function caseCounts(run: Pick<
  EvalRunRow,
  "case_pass_count" | "case_fail_count" | "case_error_count" | "case_count"
>) {
  const passed = run.case_pass_count ?? 0;
  const failed = run.case_fail_count ?? 0;
  const errored = run.case_error_count ?? 0;
  return { passed, failed, errored, judged: passed + failed, total: run.case_count ?? 0 };
}

export interface CostLabel {
  text: string;
  /** Explains a "—" or a partial sum. */
  title?: string;
  partial: boolean;
}

/**
 * A run's agent or eval cost. `null` → "—" (not traced). A value whose
 * `known` flag is false is a lower bound: some traces are missing or unpriced.
 */
export function costLabel(value: number | null | undefined, known: boolean | undefined): CostLabel {
  if (value == null) return { text: "—", title: "not traced (or no priced model call)", partial: false };
  const text = value === 0 ? "$0" : formatCost(value);
  if (known === false) {
    return { text, title: "partial: some traces are missing or unpriced", partial: true };
  }
  return { text, partial: false };
}

/** `"12 KB"`. */
export function formatBytes(n: number | null | undefined): string {
  if (typeof n !== "number" || !Number.isFinite(n)) return "? B";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export function isMediaDescriptor(v: unknown): v is EvalMediaDescriptor {
  if (!v || typeof v !== "object" || Array.isArray(v)) return false;
  const o = v as Record<string, unknown>;
  return (o.type === "image" || o.type === "pdf") && typeof o.media_idx === "number";
}

export function isBytesDescriptor(v: unknown): v is EvalBytesDescriptor {
  if (!v || typeof v !== "object" || Array.isArray(v)) return false;
  const o = v as Record<string, unknown>;
  return o.type === "bytes" && typeof o.size_bytes === "number" && Object.keys(o).length === 2;
}

/** True when a case value holds an image/PDF/bytes descriptor anywhere. */
export function containsDescriptor(v: unknown, depth = 0): boolean {
  if (depth > 6 || v == null || typeof v !== "object") return false;
  if (isMediaDescriptor(v) || isBytesDescriptor(v)) return true;
  const values = Array.isArray(v) ? v : Object.values(v as Record<string, unknown>);
  return values.some((x) => containsDescriptor(x, depth + 1));
}

/** One-line text for a descriptor: "image · 12 KB · sha256 abc123…". */
export function describeDescriptor(d: EvalMediaDescriptor | EvalBytesDescriptor): string {
  if (d.type === "bytes") return `bytes · ${formatBytes(d.size_bytes)}`;
  const parts: string[] = [d.type, formatBytes(d.size_bytes)];
  if (d.sha256) parts.push(`sha256 ${d.sha256.slice(0, 6)}…`);
  return parts.join(" · ");
}

/**
 * A case value as stable text for an equality check: each image/PDF/bytes
 * descriptor is reduced to its type, sha256 and size, so two runs' copies of
 * the same image compare equal whatever their `media_idx` or source fields.
 */
export function comparableText(value: unknown): string {
  try {
    return JSON.stringify(value ?? null, (_k, v) => {
      if (isMediaDescriptor(v)) return `[${v.type} sha256:${v.sha256 ?? "?"} size:${v.size_bytes ?? "?"}]`;
      if (isBytesDescriptor(v)) return `[bytes size:${v.size_bytes}]`;
      return v;
    });
  } catch {
    return String(value);
  }
}

/**
 * A case value as short text, with descriptors replaced by their one-line
 * description (never the raw descriptor JSON).
 */
export function caseValueText(value: unknown, max = 120): string {
  if (value == null) return "—";
  let s: string;
  if (typeof value === "string") {
    s = value;
  } else if (isMediaDescriptor(value) || isBytesDescriptor(value)) {
    s = `[${describeDescriptor(value)}]`;
  } else {
    try {
      s = JSON.stringify(value, (_k, v) =>
        isMediaDescriptor(v) || isBytesDescriptor(v) ? `[${describeDescriptor(v)}]` : v
      );
    } catch {
      s = String(value);
    }
  }
  return s.length > max ? `${s.slice(0, max)}…` : s;
}

export interface TrendRow {
  ts: number;
  series: string;
  /** Cases rate, 0..100. */
  rate: number;
  /** Checks rate, 0..100, or null. */
  checks: number | null;
  gate: string | null;
  runName: string | null;
}

/**
 * Trend points as chart rows: one row per run that reached a case verdict
 * (a null cases rate is skipped, never drawn as 0), and the series to draw,
 * the `max` with the most points first.
 */
export function buildTrend(
  points: EvalTrendPoint[],
  max = 5
): { rows: TrendRow[]; series: string[] } {
  const rows: TrendRow[] = [];
  const counts = new Map<string, number>();
  for (const p of points) {
    const rate = p.case_pass_rate;
    const ts = new Date(p.started_at).getTime();
    if (typeof rate !== "number" || !Number.isFinite(rate) || !Number.isFinite(ts)) continue;
    // An invalid run is an outage, not a quality reading: leave a gap rather
    // than a dip (the runs table still lists it, with its INVALID badge).
    if (p.gate_outcome === "invalid") continue;
    const series = p.series || p.dataset_name || p.run_name || "(unnamed)";
    counts.set(series, (counts.get(series) ?? 0) + 1);
    rows.push({
      ts,
      series,
      rate: Math.round(rate * 100),
      checks: typeof p.pass_rate === "number" ? Math.round(p.pass_rate * 100) : null,
      gate: p.gate_outcome ?? null,
      runName: p.run_name ?? null,
    });
  }
  const series = [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, max)
    .map(([name]) => name);
  const keep = new Set(series);
  return { rows: rows.filter((r) => keep.has(r.series)).sort((a, b) => a.ts - b.ts), series };
}

/** A dataset check runs an echo agent: its numbers are not model quality. */
export function isEchoRun(run: Pick<EvalRunRow, "kind" | "agent_name">): boolean {
  return run.kind === "ui-echo" || run.agent_name === "editor-echo";
}

/** Readable label for a run kind. */
export const KIND_LABEL: Record<string, string> = {
  evaluate: "evaluate",
  cli: "CLI",
  pytest: "pytest",
  optimize: "AutoLLM",
  ui: "UI",
  "ui-echo": "dataset check",
};
