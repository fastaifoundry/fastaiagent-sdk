import { useNavigate } from "react-router-dom";
import { Copy, Star } from "lucide-react";
import { toast } from "sonner";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { TraceStatusBadge } from "./TraceStatusBadge";
import { api } from "@/lib/api";
import { copyToClipboard, formatCost, formatDurationMs, formatTimeAgo, formatTokens, shortTraceId } from "@/lib/format";
import type { TraceRow } from "@/lib/types";

interface Props {
  rows: TraceRow[];
}

export function TracesTable({ rows }: Props) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const favorite = useMutation({
    mutationFn: (traceId: string) =>
      api.post<{ favorited: boolean }>(`/traces/${traceId}/favorite`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["traces"] });
    },
  });

  return (
    <div className="rounded-md border bg-card">
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead className="w-[110px]">Trace</TableHead>
            <TableHead>Name</TableHead>
            <TableHead className="w-[130px]">Agent</TableHead>
            <TableHead className="w-[90px]">Status</TableHead>
            <TableHead className="w-[80px] text-right">Spans</TableHead>
            <TableHead className="w-[90px] text-right">Duration</TableHead>
            <TableHead className="w-[80px] text-right">Tokens</TableHead>
            <TableHead className="w-[90px] text-right">Cost</TableHead>
            <TableHead className="w-[100px]">Started</TableHead>
            <TableHead className="w-[64px]"></TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => (
            <TableRow
              key={row.trace_id}
              className="cursor-pointer"
              onClick={() => navigate(`/traces/${row.trace_id}`)}
            >
              <TableCell className="font-mono text-xs text-muted-foreground">
                {shortTraceId(row.trace_id)}
              </TableCell>
              <TableCell className="font-medium truncate max-w-[240px]">
                {row.name || "—"}
              </TableCell>
              <TableCell className="truncate text-sm text-muted-foreground">
                {row.agent_name ?? "—"}
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
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
