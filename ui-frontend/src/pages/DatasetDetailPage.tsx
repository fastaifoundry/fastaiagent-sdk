import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  ChevronLeft,
  Copy,
  Download,
  Pencil,
  Play,
  Plus,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";
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
import {
  CaseEditorDialog,
  ConfirmDeleteCaseDialog,
} from "@/components/datasets/CaseEditorDialog";
import { CaseImportDialog } from "@/components/datasets/CaseImportDialog";
import {
  useAddCase,
  useDataset,
  useDeleteCase,
  useRunEval,
} from "@/hooks/use-datasets";
import type { DatasetCase, DatasetCaseInput } from "@/lib/types";

function previewInput(input: DatasetCaseInput): {
  text: string;
  multimodal: boolean;
} {
  if (typeof input === "string") {
    return { text: input, multimodal: false };
  }
  const textParts = input
    .filter((p) => p.type === "text")
    .map((p) => p.text ?? "")
    .join(" ");
  const hasMm = input.some((p) => p.type === "image" || p.type === "pdf");
  return { text: textParts, multimodal: hasMm };
}

function previewExpected(value: unknown): string {
  if (value == null) return "—";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function truncate(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n - 1) + "…";
}

/**
 * Dataset detail page: cases table + per-row actions, case editor
 * modal, import/export, and a one-click "Run eval" against the
 * currently registered echo agent.
 */
export function DatasetDetailPage() {
  const { name = "" } = useParams<{ name: string }>();
  const navigate = useNavigate();
  const { data, isLoading, error, refetch } = useDataset(name);

  const [editorOpen, setEditorOpen] = useState(false);
  const [caseToEdit, setCaseToEdit] = useState<DatasetCase | undefined>();
  const [confirmDelete, setConfirmDelete] = useState<number | null>(null);

  const add = useAddCase(name);
  const del = useDeleteCase(name);
  const runEval = useRunEval(name);

  const openCreate = () => {
    setCaseToEdit(undefined);
    setEditorOpen(true);
  };

  const openEdit = (c: DatasetCase) => {
    setCaseToEdit(c);
    setEditorOpen(true);
  };

  const duplicate = async (c: DatasetCase) => {
    try {
      await add.mutateAsync({
        input: c.input,
        expected_output: c.expected_output ?? undefined,
        tags: c.tags,
        metadata: c.metadata,
      });
      toast.success(`Duplicated case #${c.index}`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Duplicate failed");
    }
  };

  const handleConfirmDelete = async () => {
    if (confirmDelete == null) return;
    try {
      await del.mutateAsync(confirmDelete);
      toast.success(`Deleted case #${confirmDelete}`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Delete failed");
    } finally {
      setConfirmDelete(null);
    }
  };

  const handleRunEval = async () => {
    try {
      const result = await runEval.mutateAsync({ scorers: ["exact_match"] });
      toast.success(
        `Eval kicked off — ${result.pass_count} pass / ${result.fail_count} fail`
      );
      navigate(`/evals/${result.run_id}`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Run failed");
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        <TableSkeleton rows={6} />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="space-y-5">
        <PageHeader title={name} />
        <EmptyState
          title="Dataset not found"
          description={
            error instanceof Error
              ? error.message
              : "This dataset doesn't exist in the project's datasets directory."
          }
        />
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title={data.name}
        description={`${data.cases.length} case${data.cases.length === 1 ? "" : "s"}`}
      >
        <Link to="/datasets">
          <Button variant="ghost" size="sm">
            <ChevronLeft className="mr-1.5 h-3.5 w-3.5" />
            Back
          </Button>
        </Link>
        <Button variant="outline" size="sm" onClick={() => refetch()}>
          Refresh
        </Button>
        <CaseImportDialog datasetName={data.name} />
        <a
          href={`/api/datasets/${encodeURIComponent(data.name)}/export`}
          download={`${data.name}.jsonl`}
        >
          <Button variant="outline" size="sm">
            <Download className="mr-1.5 h-3.5 w-3.5" />
            Export
          </Button>
        </a>
        <Button
          variant="outline"
          size="sm"
          onClick={handleRunEval}
          disabled={runEval.isPending || data.cases.length === 0}
        >
          <Play className="mr-1.5 h-3.5 w-3.5" />
          Run eval
        </Button>
        <Button size="sm" onClick={openCreate}>
          <Plus className="mr-1.5 h-3.5 w-3.5" />
          Add case
        </Button>
      </PageHeader>

      {data.cases.length === 0 ? (
        <EmptyState
          title="No cases yet"
          description="Click 'Add case' to create one inline, or 'Import cases' to upload a JSONL file."
        />
      ) : (
        <div className="rounded-md border bg-card">
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="w-[48px] text-right">#</TableHead>
                <TableHead>Input</TableHead>
                <TableHead>Expected</TableHead>
                <TableHead className="w-[160px]">Tags</TableHead>
                <TableHead className="w-[160px] text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.cases.map((c) => {
                const preview = previewInput(c.input);
                return (
                  <TableRow
                    key={c.index}
                    className="cursor-pointer"
                    onClick={() => openEdit(c)}
                  >
                    <TableCell className="text-right font-mono text-xs text-muted-foreground">
                      {c.index}
                    </TableCell>
                    <TableCell className="max-w-[360px] font-mono text-xs">
                      {truncate(preview.text || "—", 100)}
                      {preview.multimodal && (
                        <span className="ml-2 rounded-md border bg-muted px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                          image
                        </span>
                      )}
                    </TableCell>
                    <TableCell className="max-w-[280px] font-mono text-xs">
                      {truncate(previewExpected(c.expected_output), 100)}
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {c.tags.length === 0 ? "—" : c.tags.join(", ")}
                    </TableCell>
                    <TableCell
                      className="text-right"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <div className="flex items-center justify-end gap-0.5">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7"
                          title="Edit"
                          onClick={() => openEdit(c)}
                        >
                          <Pencil className="h-3.5 w-3.5" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7"
                          title="Duplicate"
                          onClick={() => duplicate(c)}
                        >
                          <Copy className="h-3.5 w-3.5" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 text-muted-foreground hover:text-destructive"
                          title="Delete"
                          onClick={() => setConfirmDelete(c.index)}
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      )}

      <CaseEditorDialog
        datasetName={data.name}
        open={editorOpen}
        onOpenChange={setEditorOpen}
        caseToEdit={caseToEdit}
      />
      <ConfirmDeleteCaseDialog
        open={confirmDelete !== null}
        onOpenChange={(open) => {
          if (!open) setConfirmDelete(null);
        }}
        onConfirm={handleConfirmDelete}
        index={confirmDelete}
      />
    </div>
  );
}
