/**
 * The agent's guardrails: the rules it declares, then the rules that ran for
 * it without being declared (1.89.0) — a plane-distributed rule is
 * deliberately absent from ``agent.guardrails``, but its event rows saw it
 * run. Each row opens the events list filtered to that rule.
 */
import { Link } from "react-router-dom";
import { ChevronRight } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { EmptyState } from "@/components/shared/EmptyState";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { useAgentDependencies } from "@/hooks/use-agents";
import { formatTimeAgo } from "@/lib/format";
import { guardrailEventsHref } from "@/lib/guardrail";

interface Row {
  key: string;
  name: string;
  guardrail_type: string | null;
  position: string | null;
  observed: boolean;
  action?: string | null;
  blocking?: boolean | null;
  on_error?: string | null;
  severity?: string | null;
  floor?: boolean | null;
  origin?: string | null;
  checks?: number;
  last_seen?: string | null;
}

export function AgentGuardrailsSection({ agentName }: { agentName: string }) {
  const deps = useAgentDependencies(agentName);

  if (deps.isLoading) return <TableSkeleton rows={3} />;

  const rows: Row[] = [
    ...(deps.data?.guardrails ?? []).map((g, i) => ({
      key: `declared:${g.position}:${g.name}:${i}`,
      name: g.name ?? "guardrail",
      guardrail_type: g.guardrail_type,
      position: g.position,
      observed: false,
      action: g.action,
      blocking: g.blocking,
      on_error: g.on_error,
      severity: g.severity,
      floor: g.floor,
    })),
    ...(deps.data?.observed_guardrails ?? []).map((g, i) => ({
      key: `observed:${g.position}:${g.name}:${i}`,
      name: g.name,
      guardrail_type: g.guardrail_type,
      position: g.position,
      observed: true,
      origin: g.origin,
      checks: g.checks,
      last_seen: g.last_seen,
    })),
  ];

  if (rows.length === 0) {
    return (
      <EmptyState
        title="No guardrails"
        description="This agent declares no guardrails, and none have been recorded running for it."
      />
    );
  }

  return (
    <div className="rounded-md border bg-card" data-testid="agent-guardrails">
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead>Rule</TableHead>
            <TableHead className="w-[120px]">Type</TableHead>
            <TableHead className="w-[100px]">Position</TableHead>
            <TableHead className="w-[110px]">Action</TableHead>
            <TableHead className="w-[110px]">Mode</TableHead>
            <TableHead className="w-[90px]">on_error</TableHead>
            <TableHead className="w-[90px]">Severity</TableHead>
            <TableHead>Source</TableHead>
            <TableHead className="w-[40px]" />
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((r) => (
            <TableRow key={r.key}>
              <TableCell className="font-mono text-xs">
                <Link
                  to={guardrailEventsHref(agentName, r.name, r.position)}
                  className="hover:text-primary"
                >
                  {r.name}
                </Link>
                {r.observed && (
                  <span
                    className="ml-1.5 rounded-sm border border-dashed border-muted-foreground/50 px-1 py-px text-[9px] uppercase text-muted-foreground"
                    title="Ran for this agent but is not declared on it (e.g. distributed by a control plane)"
                  >
                    observed
                  </span>
                )}
                {r.floor && (
                  <span
                    className="ml-1 rounded-sm bg-primary/10 px-1 py-px text-[9px] uppercase text-primary"
                    title="Organisation baseline — only a domain admin may change this rule"
                  >
                    floor
                  </span>
                )}
              </TableCell>
              <TableCell className="text-xs text-muted-foreground">
                {r.guardrail_type ?? "—"}
              </TableCell>
              <TableCell className="text-xs text-muted-foreground">
                {r.position ?? "—"}
              </TableCell>
              <TableCell className="font-mono text-xs">{r.action ?? "—"}</TableCell>
              <TableCell className="font-mono text-xs">
                {r.blocking == null ? "—" : r.blocking ? "blocking" : "observe-only"}
              </TableCell>
              <TableCell className="font-mono text-xs">{r.on_error ?? "—"}</TableCell>
              <TableCell className="font-mono text-xs">{r.severity ?? "—"}</TableCell>
              <TableCell className="text-xs text-muted-foreground">
                {r.observed ? (
                  <>
                    {r.origin ?? "unknown origin"}
                    {r.checks != null && ` · ${r.checks} check${r.checks === 1 ? "" : "s"}`}
                    {r.last_seen && ` · ${formatTimeAgo(r.last_seen)}`}
                  </>
                ) : (
                  "declared"
                )}
              </TableCell>
              <TableCell>
                <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
