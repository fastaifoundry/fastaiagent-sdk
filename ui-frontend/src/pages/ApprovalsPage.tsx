import { useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/shared/EmptyState";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { ApprovalRow } from "@/components/approvals/ApprovalRow";
import { usePendingInterrupts } from "@/hooks/use-approvals";

export function ApprovalsPage() {
  const { data, isLoading, isFetching, refetch } = usePendingInterrupts();
  const [search, setSearch] = useState("");
  const [reasonFilter, setReasonFilter] = useState("");

  const filtered = useMemo(() => {
    const items = data?.items ?? [];
    const q = search.trim().toLowerCase();
    const r = reasonFilter.trim().toLowerCase();
    return items.filter((row) => {
      if (
        q &&
        !row.execution_id.toLowerCase().includes(q) &&
        !row.chain_name.toLowerCase().includes(q) &&
        !(row.agent_path || "").toLowerCase().includes(q)
      ) {
        return false;
      }
      if (r && !row.reason.toLowerCase().includes(r)) return false;
      return true;
    });
  }, [data?.items, search, reasonFilter]);

  const totalShown = filtered.length;
  const totalAll = data?.items.length ?? 0;

  return (
    <div className="space-y-5">
      <PageHeader
        title="Approvals"
        description={
          data
            ? `${totalAll.toLocaleString()} pending approval${
                totalAll === 1 ? "" : "s"
              }${
                search || reasonFilter
                  ? ` · ${totalShown} match filter`
                  : ""
              }`
            : undefined
        }
      >
        <Button
          variant="outline"
          size="sm"
          onClick={() => refetch()}
          disabled={isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${isFetching ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </PageHeader>

      {/* Filter bar — client-side substring match. No backend round-trip. */}
      <div className="flex flex-wrap gap-2">
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search execution_id / chain / agent_path…"
          className="h-9 flex-1 min-w-60 rounded-md border border-input bg-background px-3 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
        />
        <input
          value={reasonFilter}
          onChange={(e) => setReasonFilter(e.target.value)}
          placeholder="Filter by reason…"
          className="h-9 w-56 rounded-md border border-input bg-background px-3 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
        />
      </div>

      {isLoading ? (
        <TableSkeleton rows={5} />
      ) : totalShown === 0 ? (
        totalAll === 0 ? (
          <EmptyState
            title="No pending approvals"
            description="Workflows that call interrupt() show up here as soon as they suspend."
          />
        ) : (
          <EmptyState
            title="No approvals match these filters"
            description="Try clearing the search or reason filter."
          />
        )
      ) : (
        <div className="overflow-x-auto rounded-md border border-border">
          <table className="w-full text-left">
            <thead className="bg-muted/40">
              <tr>
                <th className="py-2 px-3 text-xs font-mono uppercase tracking-widest text-muted-foreground">
                  execution_id
                </th>
                <th className="py-2 px-3 text-xs font-mono uppercase tracking-widest text-muted-foreground">
                  chain_name
                </th>
                <th className="py-2 px-3 text-xs font-mono uppercase tracking-widest text-muted-foreground">
                  reason
                </th>
                <th className="py-2 px-3 text-xs font-mono uppercase tracking-widest text-muted-foreground">
                  agent_path
                </th>
                <th className="py-2 px-3 text-xs font-mono uppercase tracking-widest text-muted-foreground">
                  age
                </th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((p) => (
                <ApprovalRow key={p.execution_id} pending={p} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
