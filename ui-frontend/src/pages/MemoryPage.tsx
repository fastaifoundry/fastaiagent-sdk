import { useState } from "react";
import { Link } from "react-router-dom";
import { Brain, EyeOff, History, RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import {
  useLearnedMemory,
  useLearnedMemoryScopes,
} from "@/hooks/use-learned-memory";

function formatEpoch(seconds: number): string {
  if (!seconds) return "—";
  try {
    return new Date(seconds * 1000).toLocaleString();
  } catch {
    return "—";
  }
}

/**
 * Memory page — browses the ``learned_memory`` table that
 * ``PersistentFactBlock`` reads back into agents across runs. Facts are
 * produced offline by ``fastaiagent learn``; this view audits what will be
 * injected. Per-turn live memory (what a block recalled this turn, with
 * scores) lives in the trace detail's ``memory.read`` spans, not here.
 */
export function MemoryPage() {
  // "all" or an index into the scopes list. Memory facts are partitioned by
  // (scope, scope_id) where scope is user | project | agent — so the dropdown
  // lists every partition, not just agents.
  const [selected, setSelected] = useState<string>("all");
  const [redact, setRedact] = useState(false);
  const [showSuperseded, setShowSuperseded] = useState(false);

  const scopes = useLearnedMemoryScopes();
  const options = scopes.data?.scopes ?? [];
  const active = selected === "all" ? null : options[Number(selected)] ?? null;

  const facts = useLearnedMemory(
    {
      scope: active?.scope,
      scope_id: active?.scope_id,
      include_superseded: showSuperseded,
    },
    redact
  );

  const rows = facts.data?.rows ?? [];
  const totalAll = options.reduce((sum, o) => sum + o.n, 0);

  function labelFor(scope: string, scopeId: string): string {
    return scopeId ? `${scope}:${scopeId}` : scope;
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="Memory"
        description={
          facts.data
            ? `${facts.data.total} learned fact${facts.data.total === 1 ? "" : "s"} — durable memory re-injected by PersistentFactBlock`
            : undefined
        }
      >
        <label
          className="ml-1 flex items-center gap-2 rounded-md border px-2.5 py-1 text-xs font-mono"
          title="Include facts that have been replaced by a newer version (the audit history)."
        >
          <History className="h-3.5 w-3.5 text-muted-foreground" />
          <span>Show superseded</span>
          <Switch
            checked={showSuperseded}
            onCheckedChange={setShowSuperseded}
            aria-label="Toggle superseded facts"
            size="sm"
          />
        </label>
        <label
          className="flex items-center gap-2 rounded-md border px-2.5 py-1 text-xs font-mono"
          title="Sends ?redact=true to the memory API. Honored only when a RedactionPolicy(mode in {read, both}) is installed."
        >
          <EyeOff className="h-3.5 w-3.5 text-muted-foreground" />
          <span>Mask secrets</span>
          <Switch
            checked={redact}
            onCheckedChange={setRedact}
            aria-label="Toggle fact redaction"
            size="sm"
          />
        </label>
        <Button
          variant="outline"
          size="sm"
          onClick={() => facts.refetch()}
          disabled={facts.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${facts.isFetching ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </PageHeader>

      {options.length > 0 && (
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
            Scope
          </span>
          <Select value={selected} onValueChange={setSelected}>
            <SelectTrigger className="h-8 w-72 text-xs">
              <SelectValue placeholder="All scopes" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">
                All scopes
                <span className="text-muted-foreground"> ({totalAll})</span>
              </SelectItem>
              {options.map((o, i) => (
                <SelectItem key={`${o.scope}:${o.scope_id}`} value={String(i)}>
                  {labelFor(o.scope, o.scope_id)}
                  <span className="text-muted-foreground"> ({o.n})</span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      {facts.isLoading ? (
        <TableSkeleton rows={4} />
      ) : rows.length === 0 ? (
        <EmptyState
          title="No learned facts yet"
          icon={Brain}
          description="Run `fastaiagent learn` over your traces to extract durable facts, then refresh. PersistentFactBlock reads them back into agents across runs."
        />
      ) : (
        <div className="rounded-md border bg-card">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Fact</TableHead>
                <TableHead className="w-40">Scope</TableHead>
                <TableHead className="w-24">Source</TableHead>
                <TableHead className="w-24">Confidence</TableHead>
                <TableHead className="w-48">Created</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((f) => {
                const superseded = f.superseded_by != null;
                return (
                  <TableRow key={f.id} className={superseded ? "opacity-55" : undefined}>
                    <TableCell className="text-sm">
                      {f.fact}
                      {superseded && (
                        <span className="ml-2 rounded bg-muted px-1.5 py-0.5 text-[10px] font-mono text-muted-foreground">
                          superseded → #{f.superseded_by}
                        </span>
                      )}
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className="font-mono text-[11px]">
                        {f.scope_id ? `${f.scope}:${f.scope_id}` : f.scope}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-[11px]">
                      {f.source_trace_id ? (
                        <Link
                          to={`/traces/${f.source_trace_id}`}
                          className="font-mono text-primary hover:underline"
                          title={f.source_trace_id}
                        >
                          trace
                        </Link>
                      ) : (
                        <span className="font-mono text-muted-foreground">manual</span>
                      )}
                    </TableCell>
                    <TableCell className="font-mono text-xs tabular-nums">
                      {f.confidence?.toFixed(2)}
                    </TableCell>
                    <TableCell className="font-mono text-[11px] text-muted-foreground">
                      {formatEpoch(f.created_at)}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}
