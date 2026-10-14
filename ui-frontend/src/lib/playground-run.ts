/**
 * A Playground run as it was sent (1.89.0, P6/P10d/P10e).
 *
 * "Save as eval case" used to read the *current form*, so editing the prompt
 * after a run paired the old output with the new input. Each run now keeps a
 * snapshot of its own request; saving, the Response card and restoring a
 * history entry all read that, never the form.
 */
import type {
  PlaygroundDoneMetadata,
  PlaygroundParameters,
  PlaygroundPromptRef,
  SaveAsEvalRequest,
} from "@/lib/types";

/** Images up to this size are kept with the run and restored from history. */
export const MAX_KEPT_IMAGE_BYTES = 5 * 1024 * 1024;

export interface PlaygroundImage {
  name: string;
  media_type: string;
  /** Null when the image was over {@link MAX_KEPT_IMAGE_BYTES}: not kept. */
  b64: string | null;
}

export interface PlaygroundRunSnapshot {
  provider: string;
  model: string;
  prompt_template: string;
  variables: Record<string, string>;
  system_prompt: string | null;
  /** As sent: after the model's sampling rule was applied. */
  parameters: PlaygroundParameters;
  /** The endpoint, if any. The token is never part of a snapshot. */
  base_url: string | null;
  prompt_ref: PlaygroundPromptRef | null;
  /** The template with its variables filled in: the case's input. */
  resolved_prompt: string;
  image: PlaygroundImage | null;
}

export interface PlaygroundHistoryEntry {
  id: string;
  timestamp: string;
  request: PlaygroundRunSnapshot;
  response: string;
  metadata: PlaygroundDoneMetadata;
}

const VAR_RE = /\{\{(\w+)\}\}/g;

/** `{{name}}` placeholders in order of first appearance. */
export function extractVars(template: string): string[] {
  return Array.from(new Set(Array.from(template.matchAll(VAR_RE), (m) => m[1])));
}

export function resolveTemplate(template: string, values: Record<string, string>): string {
  let out = template;
  for (const [key, value] of Object.entries(values)) {
    out = out.split(`{{${key}}}`).join(value);
  }
  return out;
}

/**
 * `{{vars}}` a run would send unfilled: still in the text after substitution,
 * or given as blank. The same rule as the server's `_unfilled_variables`.
 */
export function unfilledVariables(template: string, values: Record<string, string>): string[] {
  const left = new Set(extractVars(resolveTemplate(template, values)));
  const inTemplate = new Set(extractVars(template));
  for (const [key, value] of Object.entries(values)) {
    if (!value.trim() && inTemplate.has(key)) left.add(key);
  }
  return Array.from(left).sort();
}

/** Decoded size of a base64 payload. */
export function base64Bytes(b64: string): number {
  const padding = b64.endsWith("==") ? 2 : b64.endsWith("=") ? 1 : 0;
  return Math.floor((b64.length * 3) / 4) - padding;
}

/** The image as a snapshot keeps it: the bytes only when small enough. */
export function snapshotImage(
  b64: string | null,
  mediaType: string | null,
  name: string | null
): PlaygroundImage | null {
  if (!b64 || !mediaType) return null;
  return {
    name: name ?? "image",
    media_type: mediaType,
    b64: base64Bytes(b64) <= MAX_KEPT_IMAGE_BYTES ? b64 : null,
  };
}

/** The save-as-eval body for one run: what was sent, what came back. */
export function saveAsEvalBody(
  run: PlaygroundHistoryEntry,
  datasetName: string,
  expected: string
): SaveAsEvalRequest {
  const { request, metadata } = run;
  const body: SaveAsEvalRequest = {
    dataset_name: datasetName,
    input: request.resolved_prompt,
    expected_output: expected,
    model: metadata.model,
    provider: metadata.provider,
    parameters: request.parameters,
  };
  if (request.system_prompt) body.system_prompt = request.system_prompt;
  if (request.image?.b64) {
    body.image_b64 = request.image.b64;
    body.image_media_type = request.image.media_type;
    body.image_name = request.image.name;
  }
  // Lineage only for a template sent as stored: an edited one isn't that version.
  if (metadata.prompt?.attributed) {
    body.prompt = { slug: metadata.prompt.slug, version: metadata.prompt.version };
  }
  if (metadata.trace_id) body.trace_id = metadata.trace_id;
  return body;
}

/** Whether two snapshots would send the same request (ignoring the image bytes). */
export function sameRequest(a: PlaygroundRunSnapshot, b: PlaygroundRunSnapshot): boolean {
  const key = (s: PlaygroundRunSnapshot) =>
    JSON.stringify([
      s.provider,
      s.model,
      s.prompt_template,
      s.resolved_prompt,
      s.system_prompt,
      s.parameters,
      s.base_url,
      s.image?.name ?? null,
    ]);
  return key(a) === key(b);
}
