import { fieldLabel, type ErrorInfo } from "@/lib/api-error";

/**
 * A failed Playground run (1.89.0, P1).
 *
 * Renders an {@link ErrorInfo}, never a raw response body: a FastAPI 422 has a
 * list `detail`, and rendering it took the whole page down with the form.
 */
export function RunErrorPanel({ info }: { info: ErrorInfo }) {
  return (
    <div className="space-y-1.5" data-testid="run-error">
      <pre className="rounded-md border border-destructive bg-destructive/5 p-3 font-mono text-xs whitespace-pre-wrap text-destructive">
        {info.message}
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
      {info.providerMessage && (
        <p className="text-xs text-muted-foreground">
          Provider{info.providerStatus ? ` (HTTP ${info.providerStatus})` : ""}:{" "}
          <span className="font-mono">{info.providerMessage}</span>
        </p>
      )}
      {info.hint && <p className="text-xs text-muted-foreground">{info.hint}</p>}
      {info.correlationId && !info.providerMessage && (
        <p className="text-xs text-muted-foreground">
          The provider&apos;s error is redacted here to avoid leaking account details. The full
          text is in the server log under correlation id{" "}
          <code className="font-mono">{info.correlationId}</code>.
        </p>
      )}
    </div>
  );
}
