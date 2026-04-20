import { Link, useParams } from "react-router-dom";
import { ChevronLeft, RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { TracesTable } from "@/components/traces/TracesTable";
import { useThread } from "@/hooks/use-analytics";
import type { TraceRow } from "@/lib/types";

export function ThreadPage() {
  const { threadId } = useParams<{ threadId: string }>();
  const thread = useThread(threadId);
  const traces: TraceRow[] = (thread.data?.traces ?? []) as TraceRow[];

  return (
    <div className="space-y-5">
      <PageHeader
        title={`Thread ${threadId ?? ""}`}
        description={
          thread.data
            ? `${traces.length} trace${traces.length === 1 ? "" : "s"} share this thread id`
            : undefined
        }
      >
        <Link to="/traces">
          <Button variant="ghost" size="sm">
            <ChevronLeft className="mr-1.5 h-3.5 w-3.5" />
            Traces
          </Button>
        </Link>
        <Button
          variant="outline"
          size="sm"
          onClick={() => thread.refetch()}
          disabled={thread.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${thread.isFetching ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </PageHeader>

      {thread.isLoading ? (
        <TableSkeleton rows={5} />
      ) : traces.length === 0 ? (
        <EmptyState
          title="No traces for this thread"
          description="Either the thread_id is wrong, or the traces were cleared."
        />
      ) : (
        <TracesTable rows={traces} />
      )}
    </div>
  );
}
