/**
 * Turn any error response body into something safe to render (1.89.0, P1).
 *
 * FastAPI's 422 `detail` is a LIST of `{type, loc, msg, input, ctx}` objects,
 * and the route errors use an object detail (`{error, correlation_id}`). The UI
 * assumed a string everywhere. The Playground stored the list as its error and
 * rendered it, React threw ("Objects are not valid as a React child"), and the
 * error boundary replaced the whole page — form included. Other pages showed
 * "[object Object]". Every response now goes through this one function, and
 * `message` is always a string.
 */

export interface FieldError {
  /** Dotted path without the `body`/`query`/`path` prefix, e.g. `parameters.max_tokens`. */
  field: string;
  message: string;
}

export interface ErrorInfo {
  message: string;
  status?: number;
  fieldErrors: FieldError[];
  correlationId?: string;
  kind?: string;
  providerStatus?: number;
  providerMessage?: string;
  hint?: string;
}

const LOCATION_PREFIXES = new Set(["body", "query", "path", "header", "cookie"]);

function fieldErrorsFrom(list: unknown[]): FieldError[] {
  const out: FieldError[] = [];
  for (const item of list) {
    if (!item || typeof item !== "object") continue;
    const entry = item as { loc?: unknown; msg?: unknown };
    // `input` is deliberately never read: it echoes the submitted value, which
    // for the Playground can be the API key itself.
    const loc = Array.isArray(entry.loc) ? entry.loc.map(String) : [];
    const path = LOCATION_PREFIXES.has(loc[0] ?? "") ? loc.slice(1) : loc;
    out.push({
      field: path.join(".") || "request",
      message: typeof entry.msg === "string" ? entry.msg : "invalid value",
    });
  }
  return out;
}

function str(value: unknown): string | undefined {
  return typeof value === "string" && value ? value : undefined;
}

/** Normalise an error body (already JSON-decoded, or null) into an {@link ErrorInfo}. */
export function normalizeErrorBody(
  body: unknown,
  status?: number,
  statusText?: string
): ErrorInfo {
  const fallback = statusText || (status ? `Request failed (${status})` : "Request failed");
  const detail =
    body && typeof body === "object" && "detail" in (body as object)
      ? (body as { detail: unknown }).detail
      : body;

  if (typeof detail === "string" && detail) {
    return { message: detail, status, fieldErrors: [] };
  }
  if (Array.isArray(detail)) {
    const fieldErrors = fieldErrorsFrom(detail);
    const message =
      fieldErrors.length === 1
        ? `${fieldErrors[0].field}: ${fieldErrors[0].message}`
        : fieldErrors.length > 1
          ? `${fieldErrors.length} fields are invalid`
          : fallback;
    return { message, status, fieldErrors };
  }
  if (detail && typeof detail === "object") {
    const d = detail as Record<string, unknown>;
    const providerStatus =
      typeof d.provider_status === "number" ? d.provider_status : undefined;
    return {
      message: str(d.message) ?? str(d.error) ?? fallback,
      status,
      fieldErrors: [],
      correlationId: str(d.correlation_id),
      kind: str(d.kind),
      providerStatus,
      providerMessage: str(d.provider_message),
      hint: str(d.hint),
    };
  }
  return { message: fallback, status, fieldErrors: [] };
}

/** Friendly labels for the Playground's request fields. */
const FIELD_LABELS: Record<string, string> = {
  base_url: "Endpoint",
  api_key: "Token",
  model: "Model",
  provider: "Provider",
  prompt_template: "Template",
  "parameters.max_tokens": "max_tokens",
  "parameters.temperature": "temperature",
  "parameters.top_p": "top_p",
};

export function fieldLabel(field: string): string {
  return FIELD_LABELS[field] ?? field;
}
