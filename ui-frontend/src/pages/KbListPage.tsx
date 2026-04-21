import { Database, RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { DirectoryCard } from "@/components/shared/DirectoryCard";
import { useKbCollections } from "@/hooks/use-kb";
import { formatTimeAgo } from "@/lib/format";

export function KbListPage() {
  const kbs = useKbCollections();
  const rows = kbs.data?.collections ?? [];

  return (
    <div className="space-y-5">
      <PageHeader
        title="Knowledge Bases"
        description={
          kbs.data
            ? `${rows.length} collection${rows.length === 1 ? "" : "s"} under ${kbs.data.root}`
            : undefined
        }
      >
        <Button
          variant="outline"
          size="sm"
          onClick={() => kbs.refetch()}
          disabled={kbs.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${kbs.isFetching ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </PageHeader>

      {kbs.isLoading ? (
        <TableSkeleton rows={3} />
      ) : rows.length === 0 ? (
        <EmptyState
          title="No LocalKB collections found"
          icon={Database}
          description={`Create one in code — LocalKB(name="docs").add("./docs") — then refresh.`}
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {rows.map((kb) => (
            <DirectoryCard
              key={kb.name}
              to={`/kb/${encodeURIComponent(kb.name)}`}
              icon={Database}
              title={kb.name}
              stats={[
                { label: "Documents", value: kb.doc_count.toString() },
                { label: "Chunks", value: kb.chunk_count.toString() },
                { label: "Size", value: formatBytes(kb.size_bytes) },
                { label: "Updated", value: formatTimeAgo(kb.last_updated) },
              ]}
              footer={
                <div className="truncate font-mono text-[11px] text-muted-foreground">
                  {kb.path}
                </div>
              }
            />
          ))}
        </div>
      )}
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}
