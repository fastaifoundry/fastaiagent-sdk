import { Link } from "react-router-dom";
import type { PendingInterrupt } from "@/hooks/use-approvals";

interface ApprovalRowProps {
  pending: PendingInterrupt;
}

/** Format a "/foo:a/bar:b" agent_path as a stacked breadcrumb. */
function AgentPathBreadcrumb({ path }: { path: string | null }) {
  if (!path) return <span className="text-muted-foreground">—</span>;
  const parts = path.split("/");
  return (
    <span className="font-mono text-[11px] text-muted-foreground">
      {parts.map((p, i) => (
        <span key={i}>
          {i > 0 && <span className="mx-1 text-border">›</span>}
          {p}
        </span>
      ))}
    </span>
  );
}

/** "2 hours ago"-style relative time from an ISO timestamp. */
function formatAge(iso: string): string {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return iso;
  const diff = Date.now() - then;
  const secs = Math.floor(diff / 1000);
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

export function ApprovalRow({ pending }: ApprovalRowProps) {
  return (
    <tr className="border-b border-border hover:bg-muted/30">
      <td className="py-2 px-3 font-mono text-xs">
        <Link
          to={`/approvals/${pending.execution_id}`}
          className="text-primary hover:underline"
        >
          {pending.execution_id}
        </Link>
      </td>
      <td className="py-2 px-3 text-sm">{pending.chain_name}</td>
      <td className="py-2 px-3">
        <span className="rounded bg-yellow-500/10 px-2 py-0.5 text-xs font-medium text-yellow-700 dark:text-yellow-400">
          {pending.reason}
        </span>
      </td>
      <td className="py-2 px-3">
        <AgentPathBreadcrumb path={pending.agent_path} />
      </td>
      <td className="py-2 px-3 text-xs text-muted-foreground">
        {formatAge(pending.created_at)}
      </td>
    </tr>
  );
}
