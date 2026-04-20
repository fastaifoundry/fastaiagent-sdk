import { cn } from "@/lib/utils";

interface Props {
  status: string;
  className?: string;
}

/**
 * Langfuse-style compact status pill: colored dot + uppercase label, tabular.
 */
export function TraceStatusBadge({ status, className }: Props) {
  const norm = status?.toUpperCase() || "OK";
  const meta =
    norm === "OK" || norm === "UNSET"
      ? { dot: "bg-fa-success", text: "text-fa-success", label: "OK" }
      : norm === "ERROR"
      ? { dot: "bg-destructive", text: "text-destructive", label: "ERROR" }
      : { dot: "bg-fa-warning", text: "text-fa-warning", label: norm };

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[10px] font-mono font-medium uppercase tracking-wider",
        className
      )}
    >
      <span className={cn("h-1.5 w-1.5 rounded-full", meta.dot)} />
      <span className={meta.text}>{meta.label}</span>
    </span>
  );
}
