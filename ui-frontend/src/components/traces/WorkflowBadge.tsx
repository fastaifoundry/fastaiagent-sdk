import { Bot, GitBranch, Network, UsersRound } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import type { RunnerType } from "@/lib/types";

interface Props {
  type: RunnerType;
  name?: string | null;
  /** Compact (default): icon + one-word label. Expanded: adds the runner name. */
  variant?: "compact" | "full";
  className?: string;
}

const META: Record<
  RunnerType,
  { label: string; icon: LucideIcon; chip: string }
> = {
  agent: {
    label: "agent",
    icon: Bot,
    chip: "bg-primary/10 text-primary",
  },
  chain: {
    label: "chain",
    icon: GitBranch,
    chip: "bg-accent/10 text-accent",
  },
  swarm: {
    label: "swarm",
    icon: Network,
    chip: "bg-fa-info/10 text-fa-info",
  },
  supervisor: {
    label: "supervisor",
    icon: UsersRound,
    chip: "bg-fa-warning/10 text-fa-warning",
  },
};

/**
 * Tiny identity pill for a trace's root runner. Distinguishes a plain
 * agent trace from a chain / swarm / supervisor workflow at a glance.
 */
export function WorkflowBadge({ type, name, variant = "compact", className }: Props) {
  const meta = META[type] ?? META.agent;
  const Icon = meta.icon;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[11px] font-mono uppercase tracking-wider",
        meta.chip,
        className
      )}
      title={name ?? meta.label}
    >
      <Icon className="h-3 w-3" />
      <span>{meta.label}</span>
      {variant === "full" && name && (
        <span className="normal-case tracking-normal opacity-70">· {name}</span>
      )}
    </span>
  );
}
