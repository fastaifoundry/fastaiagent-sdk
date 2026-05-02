/**
 * Inline PDF document card.
 *
 * Shows a paper icon, the filename (or media type), the page count badge,
 * and the size. Click → modal with the rendered PDF (when stored) or a
 * note that full data was not captured.
 */
import { useState } from "react";
import { FileText } from "lucide-react";
import { AttachmentModal } from "./AttachmentModal";

interface Props {
  filename?: string;
  pageCount?: number;
  sizeBytes?: number;
  /** "Text extracted" mode — the PDF was processed text-only, no page renders. */
  textExtracted?: boolean;
  attachment?: {
    traceId: string;
    spanId: string;
    attachmentId: string;
    hasFullData: boolean;
  };
}

function formatBytes(b?: number): string | null {
  if (b == null || !Number.isFinite(b)) return null;
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${Math.round(b / 1024)} KB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MB`;
}

export function PdfCard({
  filename,
  pageCount,
  sizeBytes,
  textExtracted,
  attachment,
}: Props) {
  const [open, setOpen] = useState(false);
  const sizeStr = formatBytes(sizeBytes);
  return (
    <>
      <button
        type="button"
        className="inline-flex items-center gap-3 rounded-md border bg-card px-3 py-2 hover:border-primary transition-colors"
        onClick={() => attachment && setOpen(true)}
        disabled={!attachment}
      >
        <FileText className="h-5 w-5 text-primary" />
        <div className="text-left">
          <div className="font-mono text-xs font-medium">
            {filename ?? "document.pdf"}
          </div>
          <div className="mt-0.5 flex items-center gap-2 font-mono text-[10px] text-muted-foreground">
            {pageCount != null ? (
              <span className="rounded bg-primary/10 px-1.5 py-0.5 text-primary">
                {pageCount} {pageCount === 1 ? "page" : "pages"}
              </span>
            ) : null}
            {sizeStr ? <span>{sizeStr}</span> : null}
            {textExtracted ? (
              <span className="rounded bg-muted px-1.5 py-0.5 uppercase tracking-widest">
                Text extracted
              </span>
            ) : null}
          </div>
        </div>
      </button>
      {attachment ? (
        <AttachmentModal
          open={open}
          onClose={() => setOpen(false)}
          traceId={attachment.traceId}
          spanId={attachment.spanId}
          attachmentId={attachment.attachmentId}
          mediaType="application/pdf"
          hasFullData={attachment.hasFullData}
          altText={filename ?? "document.pdf"}
        />
      ) : null}
    </>
  );
}
