import { AlertTriangle, Info } from "lucide-react";
import { JsonViewer } from "@/components/shared/JsonViewer";
import type { SpanEvent } from "@/lib/types";

interface Props {
  events: SpanEvent[];
}

/**
 * Formatter for the Events tab. Events on a span are OpenTelemetry-level
 * timestamped occurrences — separate from attributes. The dominant case
 * for the fastaiagent SDK is ``span.record_exception()`` which attaches an
 * event named ``"exception"`` carrying three well-known attributes:
 *   - ``exception.type``     e.g. "ValueError"
 *   - ``exception.message``  e.g. "agent produced no output"
 *   - ``exception.stacktrace`` full traceback as a multi-line string
 *
 * We render those prominently. Any other event (generic info events from
 * upstream OTel instrumentation, custom ``span.add_event()`` calls) falls
 * through to a generic renderer.
 */
export function EventsPane({ events }: Props) {
  if (!events || events.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        No events emitted. Span events are OpenTelemetry-level timestamped
        occurrences — most commonly they're auto-recorded exceptions. A clean
        run leaves this empty.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      {events.map((e, i) => (
        <EventCard key={i} event={e} />
      ))}
    </div>
  );
}

function EventCard({ event }: { event: SpanEvent }) {
  if (event.name === "exception") {
    return <ExceptionCard event={event} />;
  }
  return <GenericEventCard event={event} />;
}

function ExceptionCard({ event }: { event: SpanEvent }) {
  const attrs = event.attributes ?? {};
  const type = String(attrs["exception.type"] ?? "Exception");
  const message = String(attrs["exception.message"] ?? "");
  const stack = String(attrs["exception.stacktrace"] ?? "");
  const escaped = Boolean(attrs["exception.escaped"]);

  return (
    <div className="rounded-md border border-destructive/40 bg-destructive/5">
      <div className="flex items-center gap-2 border-b border-destructive/20 px-3 py-2">
        <AlertTriangle className="h-3.5 w-3.5 text-destructive" />
        <span className="font-mono text-xs font-semibold text-destructive">
          {type}
        </span>
        <span className="ml-auto text-[10px] font-mono text-muted-foreground">
          {formatEventTimestamp(event.timestamp)}
          {escaped ? " · escaped" : ""}
        </span>
      </div>
      {message && (
        <div className="px-3 py-2 text-xs">{message}</div>
      )}
      {stack && (
        <details className="border-t border-destructive/20">
          <summary className="cursor-pointer px-3 py-1.5 text-[10px] font-mono uppercase tracking-widest text-muted-foreground hover:text-foreground">
            Traceback
          </summary>
          <pre className="max-h-96 overflow-auto whitespace-pre-wrap bg-fa-terminal px-3 py-2 font-mono text-[11px] leading-snug text-fa-terminal-fg">
{stack}
          </pre>
        </details>
      )}
    </div>
  );
}

function GenericEventCard({ event }: { event: SpanEvent }) {
  const hasAttrs =
    event.attributes != null && Object.keys(event.attributes).length > 0;
  return (
    <div className="rounded-md border bg-background">
      <div className="flex items-center gap-2 border-b px-3 py-2">
        <Info className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="font-mono text-xs">{event.name}</span>
        <span className="ml-auto text-[10px] font-mono text-muted-foreground">
          {formatEventTimestamp(event.timestamp)}
        </span>
      </div>
      {hasAttrs && (
        <div className="p-3">
          <JsonViewer data={event.attributes} collapsed />
        </div>
      )}
    </div>
  );
}

/**
 * OTel stores event timestamps as nanosecond-epoch integers serialized as a
 * string (e.g. ``"1776722007568361000"``). Fall back to showing the raw
 * string for anything that doesn't look like that so we never crash on
 * unexpected formats.
 */
function formatEventTimestamp(raw: string): string {
  if (!raw) return "";
  if (/^\d{15,}$/.test(raw)) {
    const ms = Math.floor(Number(raw) / 1_000_000);
    if (Number.isFinite(ms) && ms > 0) {
      return new Date(ms).toISOString().replace("T", " ").replace("Z", "");
    }
  }
  return raw;
}
