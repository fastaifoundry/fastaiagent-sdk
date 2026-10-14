import { Link } from "react-router-dom";
import { GitFork } from "lucide-react";
import { shortTraceId } from "@/lib/format";
import type { SpanRow } from "@/lib/types";

/**
 * "Replay of <trace>" — shown on a rerun's trace (1.89.0).
 *
 * A replay rerun stamps its root span with ``fastaiagent.source="replay"``
 * and ``fastaiagent.replay.*`` attributes naming the trace it reran and how
 * (LLM determinism, tool policy). Without this the rerun's trace looked like
 * any other run of the agent.
 */
export function ReplayOriginBanner({ spans }: { spans: SpanRow[] }) {
  const root = spans.find((s) => !s.parent_span_id) ?? null;
  const attrs = root?.attributes ?? {};
  const sourceTraceId = attrs["fastaiagent.replay.source_trace_id"];
  if (typeof sourceTraceId !== "string" || !sourceTraceId) return null;

  const determinism = attrs["fastaiagent.replay.determinism"];
  const toolPolicy = attrs["fastaiagent.replay.tool_policy"];
  const forkStep = attrs["fastaiagent.replay.fork_step"];
  const surface = attrs["fastaiagent.replay.surface"];

  return (
    <div
      data-testid="replay-origin-banner"
      className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-md border border-primary/30 bg-primary/5 px-3 py-2 text-xs"
    >
      <span className="inline-flex items-center gap-1.5 font-medium">
        <GitFork className="h-3.5 w-3.5 text-primary" />
        Replay of{" "}
        <Link
          to={`/traces/${encodeURIComponent(sourceTraceId)}`}
          className="font-mono text-primary hover:underline"
          title={sourceTraceId}
        >
          {shortTraceId(sourceTraceId)}
        </Link>
      </span>
      {forkStep != null && forkStep !== "" && (
        <span className="font-mono text-muted-foreground">fork step {String(forkStep)}</span>
      )}
      {determinism != null && (
        <span className="font-mono text-muted-foreground">
          determinism: {String(determinism)}
        </span>
      )}
      {toolPolicy != null && (
        <span className="font-mono text-muted-foreground">
          tool policy: {String(toolPolicy)}
        </span>
      )}
      {surface != null && (
        <span className="font-mono text-muted-foreground">via {String(surface)}</span>
      )}
      <span className="text-muted-foreground">
        An experiment: left out of Overview, Analytics and Agents by default.
      </span>
    </div>
  );
}
