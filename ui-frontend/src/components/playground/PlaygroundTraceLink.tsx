import { Link } from "react-router-dom";
import { ExternalLink } from "lucide-react";
import { shortTraceId } from "@/lib/format";

interface Props {
  traceId: string | null | undefined;
  /**
   * ``trace_in_ui_db`` from the run response (1.89.0). False when the trace
   * went to a different store than this UI reads, so a link would 404.
   * Absent (an older server) is treated as true.
   */
  inUiDb?: boolean;
}

/** The "trace" link under a Playground response, or why there isn't one. */
export function PlaygroundTraceLink({ traceId, inUiDb }: Props) {
  if (!traceId) return null;
  if (inUiDb === false) {
    return (
      <span
        data-testid="playground-trace-elsewhere"
        className="inline-flex items-center gap-1"
        title={
          `Trace ${traceId} was written to the configured trace store, which ` +
          "is not the database this UI reads (build_app(db_path=...) differs " +
          "from the trace store). Open the UI on that store to see it."
        }
      >
        trace <span className="font-mono">{shortTraceId(traceId)}</span> was
        written to another store
      </span>
    );
  }
  return (
    <Link
      to={`/traces/${traceId}`}
      className="inline-flex items-center gap-1 text-primary hover:underline"
    >
      <ExternalLink className="h-3 w-3" />
      trace
    </Link>
  );
}
