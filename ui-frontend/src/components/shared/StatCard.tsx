import type { ReactNode } from "react";

interface StatCardProps {
  label: string;
  value: string;
  icon?: ReactNode;
  accent?: string;
}

export function StatCard({ label, value, icon, accent }: StatCardProps) {
  return (
    <div
      className={`rounded-xl border border-border bg-card p-4 ${accent || ""}`}
    >
      <div className="flex items-center gap-1.5">
        {icon}
        <p className="fa-section-label">{label}</p>
      </div>
      <p className="fa-stat-value mt-1">{value}</p>
    </div>
  );
}
