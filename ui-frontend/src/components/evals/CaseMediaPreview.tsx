import { useState } from "react";
import {
  describeDescriptor,
  isBytesDescriptor,
  isMediaDescriptor,
} from "@/lib/evalDisplay";
import type { EvalMediaDescriptor } from "@/lib/types";

/**
 * A case value that holds images, PDFs or raw bytes (1.89.0).
 *
 * Those are stored as descriptors (type, size, sha256), never their bytes.
 * Each image or PDF renders as its stored thumbnail, falling back to a
 * one-line description when there is none; the descriptor JSON is never
 * shown raw.
 */
export function CaseMediaPreview({
  value,
  runId,
  caseId,
}: {
  value: unknown;
  runId: string;
  caseId: string;
}) {
  return (
    <div className="flex flex-wrap items-start gap-2 text-xs" data-testid="case-media-preview">
      <Part value={value} runId={runId} caseId={caseId} depth={0} />
    </div>
  );
}

function Part({
  value,
  runId,
  caseId,
  depth,
}: {
  value: unknown;
  runId: string;
  caseId: string;
  depth: number;
}) {
  if (isMediaDescriptor(value)) {
    return <MediaThumb d={value} runId={runId} caseId={caseId} />;
  }
  if (isBytesDescriptor(value)) {
    return (
      <span className="rounded border bg-muted px-2 py-1 font-mono text-muted-foreground">
        {describeDescriptor(value)}
      </span>
    );
  }
  if (Array.isArray(value) && depth < 4) {
    return (
      <>
        {value.map((v, i) => (
          <Part key={i} value={v} runId={runId} caseId={caseId} depth={depth + 1} />
        ))}
      </>
    );
  }
  if (value && typeof value === "object" && depth < 4) {
    return (
      <>
        {Object.entries(value as Record<string, unknown>).map(([k, v]) => (
          <div key={k} className="flex items-start gap-1">
            <span className="font-mono text-muted-foreground">{k}:</span>
            <Part value={v} runId={runId} caseId={caseId} depth={depth + 1} />
          </div>
        ))}
      </>
    );
  }
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return <span className="whitespace-pre-wrap break-words font-mono">{text}</span>;
}

function MediaThumb({
  d,
  runId,
  caseId,
}: {
  d: EvalMediaDescriptor;
  runId: string;
  caseId: string;
}) {
  const [failed, setFailed] = useState(false);
  const label = describeDescriptor(d);
  if (failed) {
    return (
      <span
        className="rounded border bg-muted px-2 py-1 font-mono text-muted-foreground"
        title={d.sha256 ? `sha256 ${d.sha256}` : undefined}
      >
        {label}
      </span>
    );
  }
  return (
    <figure className="inline-flex flex-col gap-1">
      <img
        src={`/api/evals/${encodeURIComponent(runId)}/cases/${encodeURIComponent(
          caseId
        )}/media/${d.media_idx}`}
        alt={label}
        loading="lazy"
        onError={() => setFailed(true)}
        className="max-h-32 max-w-[200px] rounded border bg-muted object-contain"
      />
      <figcaption className="font-mono text-[10px] text-muted-foreground">{label}</figcaption>
    </figure>
  );
}
