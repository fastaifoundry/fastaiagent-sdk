import { Link } from "react-router-dom";
import { Database, RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { useKbCollections } from "@/hooks/use-kb";
import { formatTimeAgo } from "@/lib/format";
import { cn } from "@/lib/utils";

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
            <Link key={kb.name} to={`/kb/${encodeURIComponent(kb.name)}`}>
              <Card className="h-full transition-colors hover:border-primary">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-sm">
                    <Database className="h-4 w-4 shrink-0 text-muted-foreground" />
                    <span className="truncate">{kb.name}</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 pt-0">
                  <dl className="grid grid-cols-2 gap-2 text-xs">
                    <Stat label="Documents" value={kb.doc_count.toString()} />
                    <Stat label="Chunks" value={kb.chunk_count.toString()} />
                    <Stat label="Size" value={formatBytes(kb.size_bytes)} />
                    <Stat label="Updated" value={formatTimeAgo(kb.last_updated)} />
                  </dl>
                  <div className="truncate font-mono text-[11px] text-muted-foreground">
                    {kb.path}
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div>
      <dt className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        {label}
      </dt>
      <dd className={cn("font-mono text-sm tabular-nums", accent)}>{value}</dd>
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}
