import { useEffect, useState } from "react";
import { Check, Loader2, Save } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { ApiError } from "@/lib/api";
import { useSaveAsTest } from "@/hooks/use-replay";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  forkId: string;
  originalInput: string;
  originalOutput: string;
}

export function SaveAsTestDialog({
  open,
  onOpenChange,
  forkId,
  originalInput,
  originalOutput,
}: Props) {
  const [input, setInput] = useState(originalInput);
  const [expected, setExpected] = useState(originalOutput);
  const save = useSaveAsTest();

  useEffect(() => {
    if (open) {
      setInput(originalInput);
      setExpected(originalOutput);
    }
  }, [open, originalInput, originalOutput]);

  const handleSave = async () => {
    try {
      const { path } = await save.mutateAsync({
        forkId,
        input,
        expectedOutput: expected,
      });
      toast.success(`Saved to ${path}`);
      onOpenChange(false);
    } catch (e) {
      if (e instanceof ApiError) toast.error(e.message);
      else toast.error("Save failed");
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Save as regression test</DialogTitle>
          <DialogDescription>
            Appends one case to <code>./.fastaiagent/regression_tests.jsonl</code>,
            which <code>evaluate()</code> can pick up.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="test-input">Input</Label>
            <Textarea
              id="test-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              rows={3}
              className="font-mono text-xs"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="test-expected">Expected output</Label>
            <Textarea
              id="test-expected"
              value={expected}
              onChange={(e) => setExpected(e.target.value)}
              rows={6}
              className="font-mono text-xs"
            />
          </div>
        </div>

        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={save.isPending}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={save.isPending || !input.trim()}>
            {save.isPending ? (
              <>
                <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                Saving…
              </>
            ) : save.isSuccess ? (
              <>
                <Check className="mr-1.5 h-3.5 w-3.5" />
                Saved
              </>
            ) : (
              <>
                <Save className="mr-1.5 h-3.5 w-3.5" />
                Save test case
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
