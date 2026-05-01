/**
 * Render thumbnails for the multimodal attachments persisted on a span.
 *
 * Reads ``GET /api/traces/{trace_id}/spans/{span_id}/attachments`` for the
 * metadata list and renders each thumbnail via the binary endpoint
 * ``GET /api/traces/{trace_id}/spans/{span_id}/attachments/{attachment_id}``.
 *
 * Click a tile to open the binary in a new tab. PDFs render as an
 * image-of-page-1 thumbnail with a page-count badge — clicking opens the
 * full PDF when ``trace_full_images=True``, otherwise just the thumbnail.
 *
 * Refresh-based, no SSE — matches the rest of the UI.
 */
import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";

interface AttachmentMetadata {
  attachment_id: string;
  media_type: string;
  size_bytes: number;
  metadata: Record<string, unknown>;
  has_full_data: boolean;
  created_at: string;
}

interface ListResponse {
  attachments: AttachmentMetadata[];
}

interface Props {
  traceId: string;
  spanId: string;
}

export function AttachmentGallery({ traceId, spanId }: Props) {
  const [items, setItems] = useState<AttachmentMetadata[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .get<ListResponse>(
        `/traces/${traceId}/spans/${spanId}/attachments`
      )
      .then((resp) => {
        if (!cancelled) setItems(resp.attachments);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof ApiError ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [traceId, spanId]);

  if (error) {
    return (
      <p className="text-xs text-destructive">Failed to load attachments: {error}</p>
    );
  }
  if (items === null) {
    return <p className="text-xs text-muted-foreground">Loading attachments…</p>;
  }
  if (items.length === 0) {
    return null;
  }

  return (
    <div className="mb-3 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
      {items.map((item) => (
        <AttachmentTile
          key={item.attachment_id}
          traceId={traceId}
          spanId={spanId}
          item={item}
        />
      ))}
    </div>
  );
}

interface TileProps {
  traceId: string;
  spanId: string;
  item: AttachmentMetadata;
}

function AttachmentTile({ traceId, spanId, item }: TileProps) {
  const thumbUrl = `/api${`/traces/${traceId}/spans/${spanId}/attachments/${item.attachment_id}`}`;
  const fullUrl = `${thumbUrl}?full=1`;
  const isPdf = item.media_type === "application/pdf";
  const pageCount =
    typeof item.metadata?.page_count === "number"
      ? (item.metadata.page_count as number)
      : null;
  const sizeKb = Math.max(1, Math.round(item.size_bytes / 1024));

  return (
    <a
      href={item.has_full_data ? fullUrl : thumbUrl}
      target="_blank"
      rel="noreferrer"
      className="group relative block overflow-hidden rounded-md border bg-muted hover:border-primary"
      title={`${item.media_type} · ${sizeKb} KB`}
    >
      <img
        src={thumbUrl}
        alt={item.media_type}
        loading="lazy"
        className="h-32 w-full object-cover"
      />
      <div className="flex items-center justify-between gap-2 px-2 py-1 text-[10px] text-muted-foreground">
        <span>{item.media_type.split("/").pop()}</span>
        {isPdf && pageCount !== null ? (
          <span className="rounded bg-primary/10 px-1.5 py-0.5 font-medium text-primary">
            {pageCount} {pageCount === 1 ? "page" : "pages"}
          </span>
        ) : (
          <span>{sizeKb} KB</span>
        )}
      </div>
    </a>
  );
}
