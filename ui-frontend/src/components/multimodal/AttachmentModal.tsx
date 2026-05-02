/**
 * Full-size modal for an inline attachment thumbnail.
 *
 * Reads the same binary endpoint as ``AttachmentGallery``. If
 * ``trace_full_images`` was disabled when the SDK captured the trace,
 * the ``?full=1`` request 404s — we fall back to the thumbnail and show
 * the "Full resolution not stored" callout.
 */
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface Props {
  open: boolean;
  onClose: () => void;
  traceId: string;
  spanId: string;
  attachmentId: string;
  mediaType: string;
  hasFullData: boolean;
  altText?: string;
}

export function AttachmentModal({
  open,
  onClose,
  traceId,
  spanId,
  attachmentId,
  mediaType,
  hasFullData,
  altText,
}: Props) {
  const baseUrl = `/api/traces/${traceId}/spans/${spanId}/attachments/${attachmentId}`;
  const url = hasFullData ? `${baseUrl}?full=1` : baseUrl;
  const isImage = mediaType.startsWith("image/");

  return (
    <Dialog open={open} onOpenChange={(o) => (!o ? onClose() : null)}>
      <DialogContent className="max-w-4xl">
        <DialogHeader>
          <DialogTitle className="font-mono text-sm">{mediaType}</DialogTitle>
        </DialogHeader>
        <div className="flex max-h-[70vh] items-center justify-center overflow-auto">
          {isImage ? (
            <img
              src={url}
              alt={altText ?? mediaType}
              className="max-h-[70vh] max-w-full"
            />
          ) : (
            <iframe
              src={url}
              title={altText ?? mediaType}
              className="h-[70vh] w-full"
            />
          )}
        </div>
        {!hasFullData ? (
          <p className="mt-3 rounded bg-amber-500/10 border border-amber-500/30 px-3 py-2 text-[11px] text-amber-700 dark:text-amber-300">
            Full resolution not stored. Enable{" "}
            <code className="font-mono">trace_full_images=True</code> in your
            SDK config to capture full-resolution data.
          </p>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
