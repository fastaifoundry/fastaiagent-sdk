import { useEffect, useState } from "react";
import { ImagePlus, Trash2, X } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  useAddCase,
  useUpdateCase,
  useUploadImage,
} from "@/hooks/use-datasets";
import type {
  CaseBody,
  DatasetCase,
  DatasetCaseInput,
  DatasetCaseInputPart,
} from "@/lib/types";

interface Props {
  datasetName: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Pass the case to edit; omit (undefined) to create a new one. */
  caseToEdit?: DatasetCase;
  /** Pre-fill input/expected_output (used by Playground "Save as eval"). */
  initial?: { input?: DatasetCaseInput; expected_output?: unknown };
}

function inputAsText(input: DatasetCaseInput | undefined): string {
  if (input == null) return "";
  if (typeof input === "string") return input;
  return input
    .map((p) => (p.type === "text" ? p.text ?? "" : ""))
    .filter(Boolean)
    .join("\n");
}

function imagePartsOf(input: DatasetCaseInput | undefined): DatasetCaseInputPart[] {
  if (!Array.isArray(input)) return [];
  return input.filter((p) => p.type === "image" || p.type === "pdf");
}

function buildInput(text: string, attachments: DatasetCaseInputPart[]): DatasetCaseInput {
  if (attachments.length === 0) return text;
  const parts: DatasetCaseInputPart[] = [];
  if (text) parts.push({ type: "text", text });
  parts.push(...attachments);
  return parts;
}

/**
 * Modal for creating or editing a single eval case. Keeps multimodal
 * support cheap: text in a textarea, images attached via the upload
 * endpoint, the API path stored as a part on submit.
 */
export function CaseEditorDialog({
  datasetName,
  open,
  onOpenChange,
  caseToEdit,
  initial,
}: Props) {
  const isEditing = !!caseToEdit;
  const seedInput = caseToEdit?.input ?? initial?.input;
  const seedExpected = caseToEdit?.expected_output ?? initial?.expected_output;

  const [text, setText] = useState(inputAsText(seedInput));
  const [attachments, setAttachments] = useState<DatasetCaseInputPart[]>(
    imagePartsOf(seedInput)
  );
  const [expected, setExpected] = useState(
    typeof seedExpected === "string"
      ? seedExpected
      : seedExpected == null
        ? ""
        : JSON.stringify(seedExpected, null, 2)
  );
  const [tagsCsv, setTagsCsv] = useState((caseToEdit?.tags ?? []).join(", "));
  const [metaJson, setMetaJson] = useState(
    caseToEdit?.metadata && Object.keys(caseToEdit.metadata).length
      ? JSON.stringify(caseToEdit.metadata, null, 2)
      : ""
  );
  const [error, setError] = useState<string | null>(null);

  // Reset form whenever the dialog opens with a different target.
  useEffect(() => {
    if (!open) return;
    setText(inputAsText(seedInput));
    setAttachments(imagePartsOf(seedInput));
    setExpected(
      typeof seedExpected === "string"
        ? seedExpected
        : seedExpected == null
          ? ""
          : JSON.stringify(seedExpected, null, 2)
    );
    setTagsCsv((caseToEdit?.tags ?? []).join(", "));
    setMetaJson(
      caseToEdit?.metadata && Object.keys(caseToEdit.metadata).length
        ? JSON.stringify(caseToEdit.metadata, null, 2)
        : ""
    );
    setError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, caseToEdit?.index]);

  const upload = useUploadImage(datasetName);
  const add = useAddCase(datasetName);
  const update = useUpdateCase(datasetName);

  const handleAttach = async (file: File) => {
    setError(null);
    try {
      const result = await upload.mutateAsync(file);
      setAttachments((prev) => [
        ...prev,
        { type: "image", path: result.path },
      ]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    }
  };

  const handleSubmit = async () => {
    setError(null);
    let metadata: Record<string, unknown> = {};
    if (metaJson.trim()) {
      try {
        metadata = JSON.parse(metaJson);
        if (typeof metadata !== "object" || metadata === null || Array.isArray(metadata)) {
          throw new Error("metadata must be a JSON object");
        }
      } catch (e) {
        setError(`metadata: ${e instanceof Error ? e.message : "invalid JSON"}`);
        return;
      }
    }
    const tags = tagsCsv
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);
    const body: CaseBody = {
      input: buildInput(text, attachments),
      expected_output: expected || undefined,
      tags,
      metadata,
    };
    try {
      if (isEditing) {
        await update.mutateAsync({ index: caseToEdit!.index, body });
      } else {
        await add.mutateAsync(body);
      }
      onOpenChange(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>{isEditing ? "Edit case" : "Add case"}</DialogTitle>
          <DialogDescription>
            Stored as one JSONL line in <code>{datasetName}.jsonl</code> —
            same shape <code>Dataset.from_jsonl</code> reads.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div>
            <Label htmlFor="case-input" className="text-xs uppercase tracking-widest text-muted-foreground">
              Input
            </Label>
            <Textarea
              id="case-input"
              rows={4}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="What does the agent see?"
            />
          </div>

          <div>
            <Label className="text-xs uppercase tracking-widest text-muted-foreground">
              Attachments
            </Label>
            <div className="flex flex-wrap items-center gap-2">
              {attachments.map((att, i) => (
                <span
                  key={`${att.path}-${i}`}
                  className="inline-flex items-center gap-1 rounded-md border bg-muted px-2 py-1 font-mono text-xs"
                >
                  <ImagePlus className="h-3 w-3" />
                  {att.path}
                  <button
                    type="button"
                    aria-label="Remove attachment"
                    onClick={() =>
                      setAttachments((prev) => prev.filter((_, j) => j !== i))
                    }
                  >
                    <X className="h-3 w-3" />
                  </button>
                </span>
              ))}
              <label className="inline-flex cursor-pointer items-center gap-1 rounded-md border bg-card px-2 py-1 text-xs hover:bg-muted">
                <ImagePlus className="h-3.5 w-3.5" />
                Attach image
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) void handleAttach(file);
                    e.target.value = "";
                  }}
                />
              </label>
            </div>
          </div>

          <div>
            <Label htmlFor="case-expected" className="text-xs uppercase tracking-widest text-muted-foreground">
              Expected output
              <span className="ml-2 text-[10px] normal-case tracking-normal text-muted-foreground">
                Leave blank for LLM-as-Judge-only scoring.
              </span>
            </Label>
            <Textarea
              id="case-expected"
              rows={3}
              value={expected}
              onChange={(e) => setExpected(e.target.value)}
              placeholder="The reference answer (optional)."
            />
          </div>

          <div>
            <Label htmlFor="case-tags" className="text-xs uppercase tracking-widest text-muted-foreground">
              Tags <span className="ml-1 normal-case tracking-normal">(comma-separated)</span>
            </Label>
            <Input
              id="case-tags"
              value={tagsCsv}
              onChange={(e) => setTagsCsv(e.target.value)}
              placeholder="e.g. exact_match, edge_case"
            />
          </div>

          <details className="rounded-md border bg-card p-3">
            <summary className="cursor-pointer text-xs uppercase tracking-widest text-muted-foreground">
              Metadata (JSON, optional)
            </summary>
            <Textarea
              rows={4}
              className="mt-2 font-mono text-xs"
              value={metaJson}
              onChange={(e) => setMetaJson(e.target.value)}
              placeholder='{"source": "production_trace_abc"}'
            />
          </details>

          {error && (
            <div className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-600">
              {error}
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={add.isPending || update.isPending || upload.isPending}
          >
            {isEditing ? "Save" : "Add case"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

interface ConfirmDeleteCaseProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: () => void;
  index: number | null;
}

/**
 * Tiny confirmation dialog for the per-row Delete action. Kept here
 * (rather than in its own file) since it pairs 1:1 with the editor.
 */
export function ConfirmDeleteCaseDialog({
  open,
  onOpenChange,
  onConfirm,
  index,
}: ConfirmDeleteCaseProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Delete case #{index}?</DialogTitle>
          <DialogDescription>
            This rewrites the JSONL file. Subsequent cases re-index, so any
            URLs that hard-coded an index will shift.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button variant="destructive" onClick={onConfirm}>
            <Trash2 className="mr-1.5 h-3.5 w-3.5" />
            Delete
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
