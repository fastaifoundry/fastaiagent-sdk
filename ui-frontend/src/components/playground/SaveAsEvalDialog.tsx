import { useState } from "react";
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
import { useSaveAsEval } from "@/hooks/use-playground";
import { ApiError } from "@/lib/api";

interface Props {
  resolvedInput: string;
  actualOutput: string;
  systemPrompt: string | null;
  model: string | null;
  provider: string | null;
  disabled?: boolean;
}

export function SaveAsEvalDialog({
  resolvedInput,
  actualOutput,
  systemPrompt,
  model,
  provider,
  disabled,
}: Props) {
  const [open, setOpen] = useState(false);
  const [datasetName, setDatasetName] = useState("playground");
  const [expected, setExpected] = useState("");
  const save = useSaveAsEval();

  // Pre-fill expected output with the actual output the first time the
  // dialog opens — devs can tweak before saving. Reset on close so a new
  // run starts clean.
  const handleOpenChange = (next: boolean) => {
    setOpen(next);
    if (next) setExpected(actualOutput);
  };

  const handleSave = async () => {
    if (!datasetName.trim()) {
      toast.error("Dataset name is required");
      return;
    }
    if (!/^[A-Za-z0-9_\-]+$/.test(datasetName)) {
      toast.error("Dataset name must match [A-Za-z0-9_-]+");
      return;
    }
    try {
      const res = await save.mutateAsync({
        dataset_name: datasetName,
        input: resolvedInput,
        expected_output: expected,
        system_prompt: systemPrompt ?? undefined,
        model: model ?? undefined,
        provider: provider ?? undefined,
      });
      toast.success(
        `Saved to ${res.dataset_name}.jsonl (${res.line_count} case${res.line_count === 1 ? "" : "s"})`,
      );
      setOpen(false);
    } catch (e) {
      if (e instanceof ApiError) toast.error(e.message);
      else toast.error("Save failed");
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" disabled={disabled}>
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

        <div className="space-y-3">
          <div className="space-y-1">
            <Label htmlFor="dataset-name">Dataset name</Label>
            <Input
              id="dataset-name"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              placeholder="playground"
            />
          </div>
          <div className="space-y-1">
            <Label>Input (auto-captured)</Label>
            <Textarea
              value={resolvedInput}
              readOnly
              rows={4}
              className="font-mono text-xs"
            />
          </div>
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
          <Button onClick={handleSave} disabled={save.isPending}>
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
