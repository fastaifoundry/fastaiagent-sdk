import {
  AlertTriangle,
  Code2,
  Database,
  Globe,
  Network,
  Puzzle,
  Sparkles,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { EmptyState } from "@/components/shared/EmptyState";
import { formatDurationMs, formatTimeAgo } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { AgentToolsResponse, ToolOrigin } from "@/lib/types";

interface Props {
  data: AgentToolsResponse | undefined;
}

const ORIGIN_META: Record<
  ToolOrigin,
  { label: string; icon: LucideIcon; classes: string }
> = {
  function: {
    label: "function",
    icon: Code2,
    classes: "bg-primary/10 text-primary",
  },
  kb: {
    label: "kb",
    icon: Database,
    classes: "bg-fa-info/10 text-fa-info",
  },
  mcp: {
    label: "mcp",
    icon: Network,
    classes: "bg-accent/10 text-accent",
  },
  rest: {
    label: "rest",
    icon: Globe,
    classes: "bg-fa-warning/10 text-fa-warning",
  },
  custom: {
    label: "custom",
    icon: Puzzle,
    classes: "bg-muted text-muted-foreground",
  },
  unknown: {
    label: "unknown",
    icon: Sparkles,
    classes: "bg-destructive/10 text-destructive",
  },
};

export function AgentToolsSection({ data }: Props) {
  if (!data) return null;
  const { registered, used } = data;

  if (registered.length === 0 && used.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Tools</CardTitle>
        </CardHeader>
        <CardContent>
          <EmptyState
            title="No tool data for this agent yet"
            description="Run the agent — tool.* spans and the agent.tools attribute will land here automatically. Agent traces emitted before 0.9.4 won't have the origin metadata."
          />
        </CardContent>
      </Card>
    );
  }

  // Build unified rows: merge registered + used by name so the user sees
  // one row per tool with both perspectives.
  const byName = new Map<
    string,
    {
      name: string;
      origin: ToolOrigin;
      description?: string;
      registered: boolean;
      call_count?: number;
      error_count?: number;
      success_rate?: number;
      avg_latency_ms?: number;
      last_used?: string;
    }
  >();
  for (const r of registered) {
    byName.set(r.name, {
      name: r.name,
      origin: r.origin,
      description: r.description,
      registered: true,
    });
  }
  for (const u of used) {
    const existing = byName.get(u.name);
    byName.set(u.name, {
      name: u.name,
      origin: (existing?.origin ?? u.origin) as ToolOrigin,
      description: existing?.description,
      registered: existing?.registered ?? u.registered ?? false,
      call_count: u.call_count,
      error_count: u.error_count,
      success_rate: u.success_rate,
      avg_latency_ms: u.avg_latency_ms,
      last_used: u.last_used,
    });
  }
  const rows = Array.from(byName.values()).sort((a, b) => {
    // Used-with-calls first (sorted by call_count desc), then registered-but-unused.
    const aCalls = a.call_count ?? -1;
    const bCalls = b.call_count ?? -1;
    return bCalls - aCalls;
  });

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between text-sm">
          <span>Tools</span>
          <span className="text-xs font-normal text-muted-foreground">
            {registered.length} registered · {used.length} called
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead>Tool</TableHead>
              <TableHead className="w-[110px]">Origin</TableHead>
              <TableHead className="w-[90px]">Calls</TableHead>
              <TableHead className="w-[100px]">Success</TableHead>
              <TableHead className="w-[110px]">Avg latency</TableHead>
              <TableHead className="w-[120px]">Last used</TableHead>
              <TableHead className="w-[90px]">Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => {
              const meta = ORIGIN_META[row.origin] ?? ORIGIN_META.unknown;
              const Icon = meta.icon;
              const hasCalls = (row.call_count ?? 0) > 0;
              return (
                <TableRow key={row.name}>
                  <TableCell className="font-mono text-xs">
                    <div>{row.name}</div>
                    {row.description && (
                      <div className="mt-0.5 text-[11px] font-sans text-muted-foreground line-clamp-1">
                        {row.description}
                      </div>
                    )}
                  </TableCell>
                  <TableCell>
                    <span
                      className={cn(
                        "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[10px] font-mono uppercase tracking-widest",
                        meta.classes
                      )}
                      title={`Origin: ${meta.label}`}
                    >
                      <Icon className="h-3 w-3" />
                      {meta.label}
                    </span>
                  </TableCell>
                  <TableCell className="font-mono text-xs tabular-nums">
                    {hasCalls ? row.call_count : "—"}
                  </TableCell>
                  <TableCell className="font-mono text-xs tabular-nums">
                    {row.success_rate != null && hasCalls ? (
                      <span
                        className={cn(
                          row.success_rate >= 0.9
                            ? "text-fa-success"
                            : row.success_rate >= 0.7
                            ? "text-fa-warning"
                            : "text-destructive"
                        )}
                      >
                        {Math.round(row.success_rate * 100)}%
                      </span>
                    ) : (
                      "—"
                    )}
                  </TableCell>
                  <TableCell className="font-mono text-xs tabular-nums">
                    {row.avg_latency_ms != null && hasCalls
                      ? formatDurationMs(row.avg_latency_ms)
                      : "—"}
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {row.last_used ? formatTimeAgo(row.last_used) : "—"}
                  </TableCell>
                  <TableCell>
                    {row.registered && !hasCalls && (
                      <span
                        className="inline-flex items-center rounded-md bg-muted px-1.5 py-0.5 text-[10px] font-mono text-muted-foreground"
                        title="Registered with the agent but never called — possible dead code."
                      >
                        unused
                      </span>
                    )}
                    {!row.registered && hasCalls && (
                      <span
                        className="inline-flex items-center gap-1 rounded-md bg-destructive/10 px-1.5 py-0.5 text-[10px] font-mono text-destructive"
                        title="Called by the LLM but not in the registered tools list — likely a hallucinated name."
                      >
                        <AlertTriangle className="h-2.5 w-2.5" />
                        unregistered
                      </span>
                    )}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
