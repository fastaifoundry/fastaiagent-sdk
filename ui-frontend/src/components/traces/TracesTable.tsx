import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Copy, MessagesSquare, Star, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { TraceStatusBadge } from "./TraceStatusBadge";
import { WorkflowBadge } from "./WorkflowBadge";
import { api, ApiError } from "@/lib/api";
import { copyToClipboard, formatCost, formatDurationMs, formatTimeAgo, formatTokens, shortTraceId } from "@/lib/format";
import { useBulkDeleteTraces, useDeleteTrace } from "@/hooks/use-traces";
import type { TraceRow } from "@/lib/types";

interface Props {
  rows: TraceRow[];
  /** Hide the bulk-select header/checkboxes — used on nested lists. */
  hideBulkSelect?: boolean;
}

export function TracesTable({ rows, hideBulkSelect }: Props) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [confirmDelete, setConfirmDelete] = useState<
    { kind: "row"; traceId: string } | { kind: "bulk" } | null
  >(null);

  const favorite = useMutation({
    mutationFn: (traceId: string) =>
      api.post<{ favorited: boolean }>(`/traces/${traceId}/favorite`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["traces"] });
    },
  });

  const deleteOne = useDeleteTrace();
  const deleteMany = useBulkDeleteTraces();

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["traces"] });
    queryClient.invalidateQueries({ queryKey: ["overview"] });
    queryClient.invalidateQueries({ queryKey: ["analytics"] });
    queryClient.invalidateQueries({ queryKey: ["agents"] });
  };

  const handleRowDelete = async (traceId: string) => {
    try {
      await deleteOne.mutateAsync(traceId);
      toast.success("Trace deleted");
      setSelected((s) => {
        const next = new Set(s);
        next.delete(traceId);
        return next;
      });
      invalidate();
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : "Delete failed");
    } finally {
      setConfirmDelete(null);
    }
  };

  const handleBulkDelete = async () => {
    const ids = Array.from(selected);
    if (ids.length === 0) {
      setConfirmDelete(null);
      return;
    }
    try {
      const { deleted } = await deleteMany.mutateAsync(ids);
      toast.success(`Deleted ${deleted} trace${deleted === 1 ? "" : "s"}`);
      setSelected(new Set());
      invalidate();
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : "Delete failed");
    } finally {
      setConfirmDelete(null);
    }
  };

  const toggleRow = (traceId: string) => {
    setSelected((s) => {
      const next = new Set(s);
      if (next.has(traceId)) next.delete(traceId);
      else next.add(traceId);
      return next;
    });
  };

  const toggleAll = () => {
    if (selected.size === rows.length) {
      setSelected(new Set());
    } else {
      setSelected(new Set(rows.map((r) => r.trace_id)));
    }
  };

  const allSelected = rows.length > 0 && selected.size === rows.length;

  return (
    <div className="space-y-2">
      {!hideBulkSelect && selected.size > 0 && (
        <div className="flex items-center justify-between rounded-md border bg-card px-3 py-2 text-sm">
          <span className="font-mono">
            {selected.size} selected
          </span>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSelected(new Set())}
            >
              Clear
            </Button>
            <Button
              variant="destructive"
              size="sm"
              onClick={() => setConfirmDelete({ kind: "bulk" })}
              disabled={deleteMany.isPending}
            >
              <Trash2 className="mr-1.5 h-3.5 w-3.5" />
              Delete {selected.size}
            </Button>
          </div>
        </div>
      )}
      <div className="rounded-md border bg-card">
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              {!hideBulkSelect && (
                <TableHead className="w-[40px]">
                  <input
                    type="checkbox"
                    aria-label="Select all traces"
                    checked={allSelected}
                    onChange={toggleAll}
                    className="h-3.5 w-3.5 cursor-pointer"
                  />
                </TableHead>
              )}
              <TableHead className="w-[110px]">Trace</TableHead>
              <TableHead>Name</TableHead>
              <TableHead className="w-[110px]">Workflow</TableHead>
              <TableHead className="w-[130px]">Runner</TableHead>
              <TableHead className="w-[130px]">Thread</TableHead>
              <TableHead className="w-[90px]">Status</TableHead>
              <TableHead className="w-[80px] text-right">Spans</TableHead>
              <TableHead className="w-[90px] text-right">Duration</TableHead>
              <TableHead className="w-[80px] text-right">Tokens</TableHead>
              <TableHead className="w-[90px] text-right">Cost</TableHead>
              <TableHead className="w-[100px]">Started</TableHead>
              <TableHead className="w-[88px]"></TableHead>
            </TableRow>
          </TableHeader>
        <TableBody>
          {rows.map((row) => (
            <TableRow
              key={row.trace_id}
              className="cursor-pointer"
              onClick={() => navigate(`/traces/${row.trace_id}`)}
            >
              {!hideBulkSelect && (
                <TableCell
                  className="w-[40px]"
                  onClick={(e) => e.stopPropagation()}
                >
                  <input
                    type="checkbox"
                    aria-label={`Select trace ${row.trace_id}`}
                    checked={selected.has(row.trace_id)}
                    onChange={() => toggleRow(row.trace_id)}
                    className="h-3.5 w-3.5 cursor-pointer"
                  />
                </TableCell>
              )}
              <TableCell className="font-mono text-xs text-muted-foreground">
                {shortTraceId(row.trace_id)}
              </TableCell>
              <TableCell className="font-medium truncate max-w-[240px]">
                {row.name || "—"}
              </TableCell>
              <TableCell>
                <WorkflowBadge type={row.runner_type} name={row.runner_name} />
              </TableCell>
              <TableCell className="truncate text-sm text-muted-foreground">
                {row.runner_name ?? row.agent_name ?? "—"}
              </TableCell>
              <TableCell
                className="truncate text-xs text-muted-foreground"
                onClick={(e) => e.stopPropagation()}
              >
                {row.thread_id ? (
                  <Link
                    to={`/threads/${encodeURIComponent(row.thread_id)}`}
                    className="inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 font-mono hover:border-primary hover:text-primary"
                    title={`Open thread ${row.thread_id}`}
                  >
                    <MessagesSquare className="h-3 w-3" />
                    {row.thread_id.length > 14
                      ? `${row.thread_id.slice(0, 12)}…`
                      : row.thread_id}
                  </Link>
                ) : (
                  "—"
                )}
              </TableCell>
              <TableCell>
                <TraceStatusBadge status={row.status} />
              </TableCell>
              <TableCell className="text-right font-mono tabular-nums">
                {row.span_count}
              </TableCell>
              <TableCell className="text-right font-mono tabular-nums">
                {formatDurationMs(row.duration_ms)}
              </TableCell>
              <TableCell className="text-right font-mono tabular-nums">
                {formatTokens(row.total_tokens)}
              </TableCell>
              <TableCell className="text-right font-mono tabular-nums">
                {formatCost(row.total_cost_usd)}
              </TableCell>
              <TableCell className="text-xs text-muted-foreground">
                {formatTimeAgo(row.start_time)}
              </TableCell>
              <TableCell
                className="text-right"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex items-center gap-0.5">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={() => favorite.mutate(row.trace_id)}
                    title="Toggle favorite"
                  >
                    <Star className="h-3.5 w-3.5" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={async () => {
                      try {
                        await copyToClipboard(row.trace_id);
                        toast.success("Trace ID copied");
                      } catch {
                        toast.error("Copy failed");
                      }
                    }}
                    title="Copy trace ID"
                  >
                    <Copy className="h-3.5 w-3.5" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 text-muted-foreground hover:text-destructive"
                    onClick={() =>
                      setConfirmDelete({ kind: "row", traceId: row.trace_id })
                    }
                    title="Delete trace"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      </div>

      <AlertDialog
        open={confirmDelete !== null}
        onOpenChange={(open) => {
          if (!open) setConfirmDelete(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {confirmDelete?.kind === "bulk"
                ? `Delete ${selected.size} trace${selected.size === 1 ? "" : "s"}?`
                : "Delete this trace?"}
            </AlertDialogTitle>
            <AlertDialogDescription>
              Spans, notes, favorites, and linked guardrail events are removed
              from <code>local.db</code>. Eval cases that reference the trace
              are kept (trace_id is set to null). This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (!confirmDelete) return;
                if (confirmDelete.kind === "bulk") void handleBulkDelete();
                else void handleRowDelete(confirmDelete.traceId);
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
