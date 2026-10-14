import { AlertTriangle, HardDrive } from "lucide-react";
import { cn } from "@/lib/utils";
import type { RerunResult } from "@/lib/types";

/**
 * What a rerun did with each tool, and what that means for its answer
 * (1.89.0).
 *
 * A rerun runs only the tools you stub or allow; the rest are blocked and the
 * model gets an error result instead. When the model did call a blocked tool,
 * the rerun's answer reflects that error, not the tool's real effect — so the
 * page says so rather than letting the diff read as a behaviour change.
 */
const DISPOSITION_STYLE: Record<string, string> = {
  blocked: "border-destructive/40 bg-destructive/10 text-destructive",
  live: "border-fa-warning/40 bg-fa-warning/10 text-fa-warning",
  stub: "border-primary/40 bg-primary/10 text-primary",
  override: "border-primary/40 bg-primary/10 text-primary",
  no_function: "border-border bg-muted text-muted-foreground",
};

const DISPOSITION_HELP: Record<string, string> = {
  blocked: "Not run: the model got an error result instead",
  live: "Ran for real",
  stub: "Returned your canned response",
  override: "Replaced by a tool you supplied",
  no_function: "Allowed, but this process has no implementation for it",
};

export function RerunToolSummary({ result }: { result: RerunResult }) {
  const dispositions = Object.entries(result.tool_dispositions ?? {});
  const blockedCalls = Object.entries(result.blocked_tool_calls ?? {}).filter(
    ([, n]) => n > 0
  );

  return (
    <div className="space-y-2" data-testid="rerun-tool-summary">
      {(result.determinism || result.tool_policy) && (
        <p className="font-mono text-[11px] text-muted-foreground">
          {result.determinism && <>LLM: {result.determinism}</>}
          {result.determinism && result.tool_policy && " · "}
          {result.tool_policy && <>tool policy: {result.tool_policy}</>}
        </p>
      )}

      {dispositions.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
            Tools
          </span>
          {dispositions.map(([name, disposition]) => (
            <span
              key={name}
              data-testid={`tool-disposition-${name}`}
              title={DISPOSITION_HELP[disposition] ?? disposition}
              className={cn(
                "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 font-mono text-[11px]",
                DISPOSITION_STYLE[disposition] ?? DISPOSITION_STYLE.no_function
              )}
            >
              {name}
              <span className="opacity-70">· {disposition}</span>
            </span>
          ))}
        </div>
      )}

      {blockedCalls.length > 0 && (
        <div
          role="alert"
          className="flex items-start gap-2 rounded-md border border-fa-warning/40 bg-fa-warning/10 px-3 py-2 text-xs"
        >
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-fa-warning" />
          <span>
            The model called blocked tool{blockedCalls.length === 1 ? "" : "s"}{" "}
            {blockedCalls.map(([name, n], i) => (
              <span key={name}>
                {i > 0 && ", "}
                <span className="font-mono">{name}</span> ×{n}
              </span>
            ))}
            . They did not run, so the rerun's answer does not show their
            effect. Stub them or allow them live to see it.
          </span>
        </div>
      )}

      {result.exported === false && (
        <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <HardDrive className="h-3.5 w-3.5" />
          Local only: not counted in Analytics and not exported.
        </p>
      )}
      {result.exported === true && (
        <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <HardDrive className="h-3.5 w-3.5" />
          Exported like a real run, marked as a replay. Not counted in
          Analytics unless you include experiments.
        </p>
      )}
    </div>
  );
}
