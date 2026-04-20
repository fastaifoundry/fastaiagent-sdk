import { useState } from "react";
import { RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/shared/EmptyState";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { TraceFiltersBar } from "@/components/traces/TraceFilters";
import { TracesTable } from "@/components/traces/TracesTable";
import { useTraces } from "@/hooks/use-traces";
import type { TraceFilters } from "@/lib/types";

export function TracesPage() {
  const [filters, setFilters] = useState<TraceFilters>({
    page: 1,
    page_size: 100,
  });
  const { data, isLoading, isFetching, refetch } = useTraces(filters);

  const rows = data?.rows ?? [];

  return (
    <div className="space-y-5">
      <PageHeader
        title="Traces"
        description={
          data
            ? `${data.total.toLocaleString()} trace${data.total === 1 ? "" : "s"}`
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

      <TraceFiltersBar filters={filters} onChange={setFilters} />

      {isLoading ? (
        <TableSkeleton rows={10} />
      ) : rows.length === 0 ? (
        <EmptyState
          title="No traces match these filters"
          description="Try widening the time range or clearing the search term."
        />
      ) : (
        <TracesTable rows={rows} />
      )}
    </div>
  );
}
