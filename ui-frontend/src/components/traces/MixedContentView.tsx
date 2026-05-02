/**
 * Render a span input/output value as a stream of typed content parts.
 *
 * Three rendering paths:
 *   - plain string / dict with no recognizable content parts → JsonViewer
 *   - mixed content (text + image + PDF parts) → inline thumbnails / cards
 *     interleaved with text blocks
 *   - top-level lists of messages, each with a ``content`` array → flatten
 *     and render in message order
 */
import { Fragment } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { JsonViewer } from "@/components/shared/JsonViewer";
import { ImageThumb } from "@/components/multimodal/ImageThumb";
import { PdfCard } from "@/components/multimodal/PdfCard";

interface Props {
  value: unknown;
  /** Used to wire the modal to the right attachment endpoint, if known. */
  traceId?: string;
  spanId?: string;
  emptyLabel?: string;
}

interface ImagePart {
  kind: "image";
  src: string;
  mediaType: string;
  sizeBytes?: number;
  width?: number;
  height?: number;
  alt?: string;
}

interface PdfPart {
  kind: "pdf";
  filename?: string;
  pageCount?: number;
  sizeBytes?: number;
  textExtracted?: boolean;
}

interface TextPart {
  kind: "text";
  text: string;
}

interface UnknownPart {
  kind: "unknown";
  value: unknown;
}

type Part = ImagePart | PdfPart | TextPart | UnknownPart;

function classify(node: unknown): Part | null {
  if (typeof node === "string") return { kind: "text", text: node };
  if (!node || typeof node !== "object") return null;
  const obj = node as Record<string, unknown>;

  const typ =
    typeof obj.type === "string" ? obj.type.toLowerCase() : null;
  const media =
    typeof obj.media_type === "string" ? obj.media_type.toLowerCase() : null;

  if (typ === "text" || typ === "input_text" || typ === "output_text") {
    if (typeof obj.text === "string") return { kind: "text", text: obj.text };
  }

  if (
    typ === "image" ||
    typ === "input_image" ||
    typ === "image_url" ||
    (media && media.startsWith("image/"))
  ) {
    let src = "";
    if (typeof obj.image_url === "string") src = obj.image_url;
    else if (
      obj.image_url &&
      typeof obj.image_url === "object" &&
      typeof (obj.image_url as Record<string, unknown>).url === "string"
    ) {
      src = (obj.image_url as Record<string, unknown>).url as string;
    } else if (typeof obj.url === "string") src = obj.url;
    else if (typeof obj.data === "string") {
      const mt = media || "image/jpeg";
      src = `data:${mt};base64,${obj.data}`;
    }
    if (!src) return null;
    return {
      kind: "image",
      src,
      mediaType: media ?? "image/jpeg",
      sizeBytes:
        typeof obj.size_bytes === "number" ? (obj.size_bytes as number) : undefined,
      width:
        typeof obj.width === "number" ? (obj.width as number) : undefined,
      height:
        typeof obj.height === "number" ? (obj.height as number) : undefined,
      alt: typeof obj.alt === "string" ? (obj.alt as string) : undefined,
    };
  }

  if (
    typ === "input_pdf" ||
    typ === "pdf" ||
    typ === "document" ||
    media === "application/pdf"
  ) {
    return {
      kind: "pdf",
      filename: typeof obj.filename === "string" ? (obj.filename as string) : undefined,
      pageCount:
        typeof obj.page_count === "number" ? (obj.page_count as number) : undefined,
      sizeBytes:
        typeof obj.size_bytes === "number" ? (obj.size_bytes as number) : undefined,
      textExtracted: obj.text_extracted === true,
    };
  }

  return null;
}

function flatten(value: unknown): Part[] {
  if (value === null || value === undefined) return [];
  if (typeof value === "string") return [{ kind: "text", text: value }];

  const out: Part[] = [];

  function walk(node: unknown, depth = 0): void {
    if (node === null || node === undefined) return;
    if (depth > 6) return; // safety; no real payload nests this deep
    if (typeof node === "string") {
      // We only treat strings as text when they're directly inside a content
      // part list — top-level strings are handled by the caller. Here a bare
      // string in the middle of an object is most likely metadata.
      return;
    }
    if (Array.isArray(node)) {
      for (const item of node) walk(item, depth + 1);
      return;
    }
    if (typeof node === "object") {
      const obj = node as Record<string, unknown>;
      // 1. Is this object itself a content part? (image / pdf / text)
      const part = classify(node);
      if (part) {
        out.push(part);
        return;
      }
      // 2. Message-shaped: {role, content} → recurse into content.
      if ("content" in obj) {
        walk(obj.content, depth + 1);
        return;
      }
      // 3. Bag of attributes — recurse into each value so nested arrays
      //    (like ``gen_ai.request.messages``) get walked.
      for (const v of Object.values(obj)) walk(v, depth + 1);
      return;
    }
  }

  walk(value);
  return out;
}

/** True if the rendered parts contain at least one image or PDF. */
function hasMultimodal(parts: Part[]): boolean {
  return parts.some((p) => p.kind === "image" || p.kind === "pdf");
}

export function MixedContentView({ value, traceId, spanId, emptyLabel }: Props) {
  if (value === null || value === undefined) {
    return (
      <p className="text-xs text-muted-foreground">
        {emptyLabel ?? "No content."}
      </p>
    );
  }

  const parts = flatten(value);

  // Backwards-compatible path: no multimodal parts → fall through to the
  // existing JsonViewer so JSON-shaped attribute panes look identical to
  // before this feature shipped. Plain strings get a small text block.
  if (!hasMultimodal(parts)) {
    if (typeof value === "string") {
      return (
        <div className="prose prose-sm dark:prose-invert max-w-none text-sm">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{value}</ReactMarkdown>
        </div>
      );
    }
    if (
      Array.isArray(value) ||
      (typeof value === "object" && Object.keys(value as object).length > 0)
    ) {
      return <JsonViewer data={value as Record<string, unknown> | unknown[]} />;
    }
    return (
      <p className="text-xs text-muted-foreground">
        {emptyLabel ?? "No content."}
      </p>
    );
  }

  return (
    <div
      className="space-y-3"
      data-testid="mixed-content-view"
      data-multimodal="true"
    >
      {parts.map((part, idx) => (
        <Fragment key={idx}>{renderPart(part, traceId, spanId)}</Fragment>
      ))}
    </div>
  );
}

function renderPart(part: Part, traceId?: string, spanId?: string) {
  if (part.kind === "text") {
    return (
      <div className="prose prose-sm dark:prose-invert max-w-none text-sm">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{part.text}</ReactMarkdown>
      </div>
    );
  }
  if (part.kind === "image") {
    return (
      <ImageThumb
        src={part.src}
        alt={part.alt}
        mediaType={part.mediaType}
        sizeBytes={part.sizeBytes}
        width={part.width}
        height={part.height}
        attachment={
          traceId && spanId
            ? {
                traceId,
                spanId,
                attachmentId: "inline",
                hasFullData: false,
              }
            : undefined
        }
      />
    );
  }
  if (part.kind === "pdf") {
    return (
      <PdfCard
        filename={part.filename}
        pageCount={part.pageCount}
        sizeBytes={part.sizeBytes}
        textExtracted={part.textExtracted}
      />
    );
  }
  // Unknown: render as JSON so we don't silently drop content.
  return <JsonViewer data={part.value as Record<string, unknown>} />;
}

export { hasMultimodal as _testOnlyHasMultimodal };
