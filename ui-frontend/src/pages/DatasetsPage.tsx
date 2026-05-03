import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Plus, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/shared/EmptyState";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { DatasetTable } from "@/components/datasets/DatasetTable";
import { useCreateDataset, useDatasets } from "@/hooks/use-datasets";

const NAME_RE = /^[A-Za-z0-9_\-]+$/;

/**
 * Datasets list page. Lists every JSONL under
 * ``./.fastaiagent/datasets/`` plus actions for create / delete /
 * export. Detail page handles per-case CRUD.
 */
export function DatasetsPage() {
  const { data, isLoading, isFetching, refetch } = useDatasets();
  const [createOpen, setCreateOpen] = useState(false);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const create = useCreateDataset();
  const navigate = useNavigate();

  const handleCreate = async () => {
    setError(null);
    if (!NAME_RE.test(name)) {
      setError("Name must be letters, numbers, underscores, or dashes only.");
      return;
    }
    try {
      await create.mutateAsync(name);
      toast.success(`Created dataset '${name}'`);
      setCreateOpen(false);
      setName("");
      navigate(`/datasets/${encodeURIComponent(name)}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Create failed");
    }
  };

  const rows = data ?? [];

  return (
    <div className="space-y-5">
      <PageHeader
        title="Eval Datasets"
        description={
          data
            ? `${rows.length} dataset${rows.length === 1 ? "" : "s"}`
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
        <Button size="sm" onClick={() => setCreateOpen(true)}>
          <Plus className="mr-1.5 h-3.5 w-3.5" />
          New dataset
        </Button>
      </PageHeader>

      {isLoading ? (
        <TableSkeleton rows={6} />
      ) : rows.length === 0 ? (
        <EmptyState
          title="No datasets yet"
          description="Create one with 'New dataset', or save a Playground run via 'Save as eval case'."
        />
      ) : (
        <DatasetTable rows={rows} />
      )}

      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle>New dataset</DialogTitle>
            <DialogDescription>
              Creates an empty <code>{"{name}.jsonl"}</code> under
              <code> ./.fastaiagent/datasets/</code>.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2">
            <Label htmlFor="ds-name" className="text-xs uppercase tracking-widest text-muted-foreground">
              Name
            </Label>
            <Input
              id="ds-name"
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. refund-policy-qa"
            />
            {error && (
              <p className="text-xs text-red-600">{error}</p>
            )}
          </div>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setCreateOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleCreate}
              disabled={!name || create.isPending}
            >
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
