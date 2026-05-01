/**
 * Dialog the Export button on the Trace detail page opens.
 *
 * Two checkboxes — embed attachment bytes, embed checkpoint state —
 * compose into the URL query params on the existing
 * ``GET /api/traces/{trace_id}/export`` endpoint. The browser handles
 * the actual download via Content-Disposition.
 */
import { useState } from "react";
import { Download } from "lucide-react";
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
import { Checkbox } from "@/components/ui/checkbox";

interface Props {
  traceId: string;
}

export function ExportTraceDialog({ traceId }: Props) {
  const [open, setOpen] = useState(false);
  const [includeAttachments, setIncludeAttachments] = useState(false);
  const [includeCheckpointState, setIncludeCheckpointState] = useState(false);

  const url = (() => {
    const params = new URLSearchParams();
    if (includeAttachments) params.set("include_attachments", "true");
    if (includeCheckpointState) params.set("include_checkpoint_state", "true");
    const qs = params.toString();
    return `/api/traces/${traceId}/export${qs ? `?${qs}` : ""}`;
  })();

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" data-testid="export-trace-button">
          <Download className="mr-1.5 h-3.5 w-3.5" />
          Export
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-md" data-testid="export-trace-dialog">
        <DialogHeader>
          <DialogTitle>Export trace as JSON</DialogTitle>
          <DialogDescription>
            Download a self-contained JSON file containing the trace
            metadata, every span, and any associated checkpoints.
            Attachments default to metadata-only.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-3 py-2">
          <label className="flex items-start gap-3 text-sm">
            <Checkbox
              checked={includeAttachments}
              onCheckedChange={(v) => setIncludeAttachments(v === true)}
              data-testid="export-include-attachments"
            />
            <span>
              <span className="font-medium">Include image / PDF data</span>
              <span className="block text-xs text-muted-foreground">
                Embeds attachment bytes as base64. Files can be large —
                disable for sharing in chat or pasting into issues.
              </span>
            </span>
          </label>
          <label className="flex items-start gap-3 text-sm">
            <Checkbox
              checked={includeCheckpointState}
              onCheckedChange={(v) => setIncludeCheckpointState(v === true)}
              data-testid="export-include-checkpoint-state"
            />
            <span>
              <span className="font-medium">Include checkpoint state</span>
              <span className="block text-xs text-muted-foreground">
                Embeds the full <code className="font-mono">state_snapshot</code>{" "}
                for every checkpoint. Off by default — most state snapshots
                are large.
              </span>
            </span>
          </label>
        </div>
        <DialogFooter>
          <Button variant="ghost" size="sm" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button asChild size="sm">
            <a
              href={url}
              data-testid="export-trace-download"
              onClick={() => setOpen(false)}
            >
              <Download className="mr-1.5 h-3.5 w-3.5" />
              Download
            </a>
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
