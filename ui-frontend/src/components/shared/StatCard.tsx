import type { ReactNode } from "react";

interface StatCardProps {
  label: string;
  value: string;
  icon?: ReactNode;
  accent?: string;
}

export function StatCard({ label, value, icon, accent }: StatCardProps) {
  return (
    <div className={`rounded-md border p-4 bg-card ${accent || ""}`}>
      <div className="flex items-center gap-1.5">
        {icon}
        <p className="text-xs font-mono uppercase tracking-widest text-muted-foreground">
          {label}
        </p>
      </div>
      <p className="text-3xl font-bold font-mono mt-1">{value}</p>
    </div>
  );
}
