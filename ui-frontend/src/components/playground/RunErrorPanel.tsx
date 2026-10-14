import { fieldLabel, type ErrorInfo } from "@/lib/api-error";

/** Human labels for the server's failure kinds (`ui/llm_errors.py`, 1.89.0, P8). */
const KIND_LABELS: Record<string, string> = {
  auth: "Authentication failed",
  not_found: "Model not found",
  bad_request: "Request rejected by the provider",
  rate_limited: "Rate limited",
  provider_error: "Provider error",
  connection: "Could not connect",
  timeout: "Timed out",
  config: "Configuration problem",
  internal: "Internal error",
};

function kindLabel(kind: string): string {
  return KIND_LABELS[kind] ?? kind;
}

/**
 * A failed Playground run (1.89.0, P1/P8).
 *
 * Renders an {@link ErrorInfo}, never a raw response body: a FastAPI 422 has a
 * list `detail`, and rendering it took the whole page down with the form.
 *
 * Since P8 a provider failure says what went wrong (the kind, the provider's
 * scrubbed message, a hint). With `FASTAIAGENT_UI_PROVIDER_ERRORS=redacted`
 * those fields are absent and the opaque message plus correlation id is all
 * there is.
 */
export function RunErrorPanel({ info }: { info: ErrorInfo }) {
  return (
    <div className="space-y-1.5" data-testid="run-error">
      {info.kind && (
        <p className="text-sm font-medium text-destructive" data-testid="run-error-kind">
          {kindLabel(info.kind)}
          {info.providerStatus ? ` (HTTP ${info.providerStatus})` : ""}
        </p>
      )}
      <pre className="rounded-md border border-destructive bg-destructive/5 p-3 font-mono text-xs whitespace-pre-wrap text-destructive">
        {info.providerMessage ?? info.message}
      </pre>
      {info.fieldErrors.length > 0 && (
        <ul className="list-disc space-y-0.5 pl-5 text-xs text-destructive">
          {info.fieldErrors.map((f) => (
            <li key={`${f.field}:${f.message}`}>
              <span className="font-medium">{fieldLabel(f.field)}</span>: {f.message}
            </li>
          ))}
        </ul>
      )}
      {info.hint && <p className="text-xs text-foreground">{info.hint}</p>}
      {info.correlationId &&
        (info.providerMessage || info.kind ? (
          <p className="text-[11px] text-muted-foreground">
            Server log correlation id <code className="font-mono">{info.correlationId}</code>
          </p>
        ) : (
          <p className="text-xs text-muted-foreground">
            The provider&apos;s error text is not shown here. The full text is in the server log
            under correlation id <code className="font-mono">{info.correlationId}</code>.
          </p>
        ))}
    </div>
  );
}
