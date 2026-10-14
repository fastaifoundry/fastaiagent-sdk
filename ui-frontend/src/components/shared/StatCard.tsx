import type { ReactNode } from "react";

interface StatCardProps {
  label: string;
  value: string;
  icon?: ReactNode;
  accent?: string;
  /** A short line under the value, e.g. "16/20". */
  hint?: ReactNode;
  /** Hover text for the whole card. */
  title?: string;
  /** Hover text for the value only, e.g. why it is "—". */
  valueTitle?: string;
}

export function StatCard({ label, value, icon, accent, hint, title, valueTitle }: StatCardProps) {
  return (
    <div
      className={`rounded-xl border border-border bg-card p-4 ${accent || ""}`}
      title={title}
      data-stat={label}
    >
      <div className="flex items-center gap-1.5">
        {icon}
        <p className="fa-section-label">{label}</p>
      </div>
      <p className="fa-stat-value mt-1" title={valueTitle}>
        {value}
      </p>
      {hint != null && hint !== "" && (
        <p className="mt-0.5 font-mono text-[11px] text-muted-foreground">{hint}</p>
      )}
    </div>
  );
}
