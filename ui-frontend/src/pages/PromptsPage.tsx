import { Link } from "react-router-dom";
import { RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
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
import { RegistryExternalBanner } from "@/components/prompts/RegistryGate";
import { usePrompts } from "@/hooks/use-prompts";

export function PromptsPage() {
  const prompts = usePrompts();
  const rows = prompts.data?.rows ?? [];
  const isLocal = prompts.data?.registry_is_local ?? true;

  return (
    <div className="space-y-5">
      <PageHeader
        title="Prompts"
        description={
          prompts.data
            ? `${rows.length} prompt${rows.length === 1 ? "" : "s"} in registry`
            : undefined
        }
      >
        <Button
          variant="outline"
          size="sm"
          onClick={() => prompts.refetch()}
          disabled={prompts.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${prompts.isFetching ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </PageHeader>

      {!isLocal && <RegistryExternalBanner />}

      {prompts.isLoading ? (
        <TableSkeleton rows={5} />
      ) : rows.length === 0 ? (
        <EmptyState
          title="No prompts registered"
          description={
            isLocal
              ? "Call PromptRegistry().register(...) from code — they'll appear here."
              : "This registry is read-only from the UI."
          }
        />
      ) : (
        <div className="rounded-md border bg-card">
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead>Name</TableHead>
                <TableHead className="w-[100px] text-right">Latest</TableHead>
                <TableHead className="w-[100px] text-right">Versions</TableHead>
                <TableHead className="w-[140px] text-right">Linked traces</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => (
                <TableRow key={row.name} className="cursor-pointer">
                  <TableCell>
                    <Link
                      to={`/prompts/${encodeURIComponent(row.name)}`}
                      className="font-medium hover:text-primary"
                    >
                      {row.name}
                    </Link>
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums">
                    v{row.latest_version}
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums">
                    {row.versions}
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums text-muted-foreground">
                    {row.linked_trace_count}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}
