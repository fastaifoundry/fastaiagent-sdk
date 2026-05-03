import { useState } from "react";
import { Upload } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useImportJsonl } from "@/hooks/use-datasets";

interface Props {
  datasetName: string;
}

/**
 * Upload a JSONL file into an existing dataset. Validation happens on
 * the server — the dialog surfaces the line-numbered error message
 * verbatim so the user knows which line to fix.
 */
export function CaseImportDialog({ datasetName }: Props) {
  const [open, setOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<"append" | "replace">("append");
  const [error, setError] = useState<string | null>(null);
  const importer = useImportJsonl(datasetName);

  const handleSubmit = async () => {
    if (!file) return;
    setError(null);
    try {
      await importer.mutateAsync({ file, mode });
      setOpen(false);
      setFile(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Import failed");
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <Upload className="mr-1.5 h-3.5 w-3.5" />
          Import cases
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Import JSONL</DialogTitle>
          <DialogDescription>
            Each line must be a JSON object with at least an{" "}
            <code>input</code> field.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          <div>
            <Label className="text-xs uppercase tracking-widest text-muted-foreground">
              File
            </Label>
            <input
              type="file"
              accept=".jsonl,application/x-ndjson,application/json,text/plain"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="block w-full text-sm"
            />
          </div>

          <div>
            <Label className="text-xs uppercase tracking-widest text-muted-foreground">
              Mode
            </Label>
            <div className="flex items-center gap-2">
              {(["append", "replace"] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setMode(m)}
                  className={
                    "rounded-md border px-3 py-1 text-xs " +
                    (mode === m
                      ? "bg-primary text-primary-foreground border-primary"
                      : "bg-card hover:bg-muted")
                  }
                >
                  {m}
                </button>
              ))}
            </div>
          </div>

          {error && (
            <div className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-600">
              {error}
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="ghost" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={!file || importer.isPending}>
            Import
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
