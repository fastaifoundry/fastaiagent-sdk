/**
 * Inline image thumbnail.
 *
 * Renders an image content part inside the trace input/output panes. The
 * image is loaded directly from the URL embedded in the content part — if
 * the part comes from a traced span we also render a click-to-modal for
 * full-size viewing, but if the part is just a remote URL we let the
 * browser handle it.
 */
import { useState } from "react";
import { ZoomIn } from "lucide-react";
import { AttachmentModal } from "./AttachmentModal";

interface Props {
  src: string;
  alt?: string;
  mediaType?: string;
  sizeBytes?: number;
  width?: number;
  height?: number;
  /** When the thumbnail is backed by a stored attachment, click → modal. */
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

export function ImageThumb({
  src,
  alt,
  mediaType = "image/*",
  sizeBytes,
  width,
  height,
  attachment,
}: Props) {
  const [open, setOpen] = useState(false);
  const sizeStr = formatBytes(sizeBytes);
  const dimStr = width && height ? `${width}×${height}` : null;
  return (
    <div className="inline-block max-w-[300px]">
      <button
        type="button"
        className="group relative block overflow-hidden rounded-md border bg-muted hover:border-primary"
        onClick={() => setOpen(true)}
        aria-label="Open full size"
      >
        <img
          src={src}
          alt={alt ?? mediaType}
          loading="lazy"
          className="block max-h-[200px] max-w-[300px] object-contain"
        />
        <span className="absolute right-1 top-1 rounded bg-background/80 p-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <ZoomIn className="h-3.5 w-3.5" />
        </span>
      </button>
      <div className="mt-1 flex items-center gap-2 text-[10px] font-mono text-muted-foreground">
        <span>{mediaType}</span>
        {sizeStr ? (
          <>
            <span>·</span>
            <span>{sizeStr}</span>
          </>
        ) : null}
        {dimStr ? (
          <>
            <span>·</span>
            <span>{dimStr}</span>
          </>
        ) : null}
      </div>
      {attachment ? (
        <AttachmentModal
          open={open}
          onClose={() => setOpen(false)}
          traceId={attachment.traceId}
          spanId={attachment.spanId}
          attachmentId={attachment.attachmentId}
          mediaType={mediaType}
          hasFullData={attachment.hasFullData}
          altText={alt}
        />
      ) : open ? (
        <Dialog
          src={src}
          alt={alt ?? mediaType}
          onClose={() => setOpen(false)}
        />
      ) : null}
    </div>
  );
}

function Dialog({ src, alt, onClose }: { src: string; alt: string; onClose: () => void }) {
  return (
    <div
      role="dialog"
      aria-modal="true"
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6 cursor-zoom-out"
    >
      <img src={src} alt={alt} className="max-h-[90vh] max-w-[90vw]" />
    </div>
  );
}
