import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Download, Image as ImageIcon, Trash2 } from "lucide-react";
import { toast } from "sonner";
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
import { useDeleteDataset } from "@/hooks/use-datasets";
import type { DatasetSummary } from "@/lib/types";
import { Timestamp } from "@/components/shared/Timestamp";

interface Props {
  rows: DatasetSummary[];
}

/**
 * Datasets list view. The whole row opens the detail page for inline editing;
 * the export and delete actions live in their own cell and don't trigger it.
 *
 * The row used to be inert except for the name link — the comment here claimed
 * "each row links to the detail page" while only one cell actually did, and the
 * actions cell already stopped propagation for a row handler that was never
 * added. Every other list in the UI (traces, evals, guardrails, simulations,
 * optimizes, prompts) opens from anywhere in the row.
 */
export function DatasetTable({ rows }: Props) {
  const navigate = useNavigate();
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const del = useDeleteDataset();

  const handleDelete = async (name: string) => {
    try {
      await del.mutateAsync(name);
      toast.success(`Deleted dataset '${name}'`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Delete failed");
    } finally {
      setConfirmDelete(null);
    }
  };

  return (
    <div className="space-y-2">
      <div className="rounded-md border bg-card">
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead>Name</TableHead>
              <TableHead className="w-[80px] text-right">Cases</TableHead>
              <TableHead className="w-[80px]">Multimodal</TableHead>
              <TableHead className="w-[140px]">Last modified</TableHead>
              <TableHead className="w-[140px]">Created</TableHead>
              <TableHead className="w-[160px] text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => (
              <TableRow
                key={row.name}
                className="cursor-pointer"
                onClick={() =>
                  navigate(`/datasets/${encodeURIComponent(row.name)}`)
                }
              >
                {/* The name stays a real anchor so keyboard focus, middle-click
                    and open-in-new-tab keep working; the cell stops propagation
                    so the row handler doesn't also fire. */}
                <TableCell
                  className="font-medium"
                  onClick={(e) => e.stopPropagation()}
                >
                  <Link
                    to={`/datasets/${encodeURIComponent(row.name)}`}
                    className="hover:text-primary"
                  >
                    {row.name}
                  </Link>
                </TableCell>
                <TableCell className="text-right font-mono tabular-nums">
                  {row.case_count}
                </TableCell>
                <TableCell>
                  {row.has_multimodal && (
                    <ImageIcon
                      className="h-3.5 w-3.5 text-muted-foreground"
                      aria-label="Contains image/PDF cases"
                    />
                  )}
                </TableCell>
                <TableCell className="text-xs text-muted-foreground">
                  <Timestamp iso={row.modified_at} />
                </TableCell>
                <TableCell className="text-xs text-muted-foreground">
                  <Timestamp iso={row.created_at} />
                </TableCell>
                <TableCell
                  className="text-right"
                  onClick={(e) => e.stopPropagation()}
                >
                  <div className="flex items-center justify-end gap-1">
                    <a
                      href={`/api/datasets/${encodeURIComponent(row.name)}/export`}
                      download={`${row.name}.jsonl`}
                    >
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7"
                        title="Export as JSONL"
                      >
                        <Download className="h-3.5 w-3.5" />
                      </Button>
                    </a>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-muted-foreground hover:text-destructive"
                      onClick={() => setConfirmDelete(row.name)}
                      title="Delete dataset"
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
              Delete dataset '{confirmDelete}'?
            </AlertDialogTitle>
            <AlertDialogDescription>
              The JSONL file and any uploaded images for this dataset are
              removed from disk. Eval runs already pointing at this
              dataset are kept (their persisted snapshots remain valid).
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (confirmDelete) void handleDelete(confirmDelete);
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
