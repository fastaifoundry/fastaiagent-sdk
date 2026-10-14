import { useEffect, useState } from "react";
import { Loader2, Save } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useDatasets } from "@/hooks/use-datasets";
import { useSaveAsEval } from "@/hooks/use-playground";
import { ApiError } from "@/lib/api";
import { saveAsEvalBody, type PlaygroundHistoryEntry } from "@/lib/playground-run";

interface Props {
  /** The run to save: its own snapshot, never the current form (1.89.0, P6). */
  run: PlaygroundHistoryEntry | null;
  /** The form no longer matches the run; the dialog says which one is saved. */
  formChanged?: boolean;
  disabled?: boolean;
}

function formatParams(p: PlaygroundHistoryEntry["request"]["parameters"]): string {
  const parts = [`max_tokens=${p.max_tokens}`];
  if (p.temperature !== null) parts.push(`temperature=${p.temperature}`);
  if (p.top_p !== null) parts.push(`top_p=${p.top_p}`);
  return parts.join(" · ");
}

export function SaveAsEvalDialog({ run, formChanged, disabled }: Props) {
  const [open, setOpen] = useState(false);
  const [datasetName, setDatasetName] = useState("playground");
  const [expected, setExpected] = useState("");
  const [creatingNew, setCreatingNew] = useState(false);
  const save = useSaveAsEval();
  const datasets = useDatasets();

  // Pre-fill expected output with the run's output each time the dialog
  // opens — devs can tweak before saving. Also default the dataset to an
  // existing one (rather than a literal "playground" string) so the combo
  // lands on a sensible value the user can accept or override.
  const handleOpenChange = (next: boolean) => {
    setOpen(next);
    if (next) setExpected(run?.response ?? "");
  };

  useEffect(() => {
    if (!open) return;
    const list = datasets.data ?? [];
    if (list.length > 0 && !creatingNew) {
      // Prefer "playground" if it already exists; otherwise the first.
      const has = list.find((d) => d.name === "playground");
      setDatasetName(has ? has.name : list[0]!.name);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, datasets.data]);

  const handleSave = async () => {
    if (!run) return;
    if (!datasetName.trim()) {
      toast.error("Dataset name is required");
      return;
    }
    if (!/^[A-Za-z0-9_-]+$/.test(datasetName)) {
      toast.error("Dataset name must match [A-Za-z0-9_-]+");
      return;
    }
    try {
      const res = await save.mutateAsync(saveAsEvalBody(run, datasetName, expected));
      toast.success(
        `Saved to ${res.dataset_name}.jsonl (${res.line_count} case${res.line_count === 1 ? "" : "s"})`,
      );
      setOpen(false);
    } catch (e) {
      if (e instanceof ApiError) toast.error(e.message);
      else toast.error("Save failed");
    }
  };

  const request = run?.request;
  const image = request?.image ?? null;

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          disabled={disabled || !run}
          title={!run ? "Save needs a completed run" : undefined}
        >
          <Save className="mr-1.5 h-3.5 w-3.5" />
          Save as eval case
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Save as eval case</DialogTitle>
          <DialogDescription>
            Appends a JSONL line to{" "}
            <code className="font-mono text-xs">
              .fastaiagent/datasets/{datasetName || "{name}"}.jsonl
            </code>{" "}
            so the case is runnable via{" "}
            <code className="font-mono text-xs">Dataset.from_jsonl()</code>.
          </DialogDescription>
        </DialogHeader>

        {formChanged && (
          <p
            className="rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-xs text-amber-900 dark:text-amber-200"
            data-testid="save-eval-form-changed"
          >
            The form has changed since this run. This saves the run as it was
            sent, not the form.
          </p>
        )}

        <div className="space-y-3">
          <div className="space-y-1">
            <Label htmlFor="dataset-name">Dataset</Label>
            {!creatingNew && (datasets.data?.length ?? 0) > 0 ? (
              <div className="flex items-center gap-2">
                <select
                  id="dataset-name"
                  value={datasetName}
                  onChange={(e) => setDatasetName(e.target.value)}
                  className="h-9 flex-1 rounded-md border border-input bg-background px-2 text-sm"
                >
                  {(datasets.data ?? []).map((d) => (
                    <option key={d.name} value={d.name}>
                      {d.name} ({d.case_count})
                    </option>
                  ))}
                </select>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setCreatingNew(true);
                    setDatasetName("");
                  }}
                >
                  + New
                </Button>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <Input
                  id="dataset-name"
                  value={datasetName}
                  onChange={(e) => setDatasetName(e.target.value)}
                  placeholder="playground"
                  autoFocus={creatingNew}
                />
                {creatingNew && (datasets.data?.length ?? 0) > 0 && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => setCreatingNew(false)}
                  >
                    Use existing
                  </Button>
                )}
              </div>
            )}
          </div>
          <div className="space-y-1">
            <Label htmlFor="save-eval-input">Input (as sent)</Label>
            <Textarea
              id="save-eval-input"
              value={request?.resolved_prompt ?? ""}
              readOnly
              rows={4}
              className="font-mono text-xs"
            />
          </div>
          {request?.system_prompt && (
            <div className="space-y-1">
              <Label htmlFor="save-eval-system">System prompt (kept in metadata)</Label>
              <Textarea
                id="save-eval-system"
                value={request.system_prompt}
                readOnly
                rows={2}
                className="font-mono text-xs"
              />
            </div>
          )}
          {image && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              {image.b64 ? (
                <>
                  <img
                    src={`data:${image.media_type};base64,${image.b64}`}
                    alt={image.name}
                    className="h-10 w-10 rounded object-cover"
                  />
                  <span className="font-mono">{image.name}</span> is saved with
                  the case.
                </>
              ) : (
                <span data-testid="save-eval-image-dropped">
                  <span className="font-mono">{image.name}</span> is over 5 MB and
                  was not kept, so the case is saved without it.
                </span>
              )}
            </div>
          )}
          {run && (
            <p className="font-mono text-[11px] text-muted-foreground" data-testid="save-eval-run">
              {run.metadata.provider}/{run.metadata.model} · {formatParams(run.request.parameters)}
              {run.metadata.prompt && ` · ${run.metadata.prompt.slug}`}
              {run.metadata.prompt?.version != null && ` v${run.metadata.prompt.version}`}
            </p>
          )}
          <div className="space-y-1">
            <Label htmlFor="expected">Expected output</Label>
            <Textarea
              id="expected"
              value={expected}
              onChange={(e) => setExpected(e.target.value)}
              rows={6}
              className="font-mono text-xs"
              placeholder="Leave as the actual output, or edit to match the desired answer."
            />
          </div>
        </div>

        <DialogFooter>
          <Button variant="ghost" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={save.isPending || !run}>
            {save.isPending ? (
              <>
                <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                Saving…
              </>
            ) : (
              "Save"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
