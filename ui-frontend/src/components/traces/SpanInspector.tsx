import { useMemo } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { JsonViewer } from "@/components/shared/JsonViewer";
import { TraceStatusBadge } from "./TraceStatusBadge";
import { formatDurationMs } from "@/lib/format";
import type { SpanRow } from "@/lib/types";

interface Props {
  span: SpanRow | null;
}

const INPUT_KEYS = new Set([
  "gen_ai.request.messages",
  "gen_ai.request.prompt",
  "agent.input",
  "chain.input",
  "swarm.input",
  "supervisor.input",
  "tool.input",
  "tool.args",
  "retrieval.query",
  "input",
]);

const OUTPUT_KEYS = new Set([
  "gen_ai.response.content",
  "gen_ai.response.tool_calls",
  "agent.output",
  "chain.output",
  "swarm.output",
  "supervisor.output",
  "tool.output",
  "tool.result",
  "retrieval.doc_ids",
  "retrieval.result_count",
  "output",
]);

function pick(
  attrs: Record<string, unknown>,
  match: (k: string) => boolean
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(attrs)) {
    if (match(k)) out[k] = v;
  }
  return out;
}

function tryParseJson(value: unknown): unknown {
  if (typeof value !== "string") return value;
  const trimmed = value.trim();
  if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return value;
  try {
    return JSON.parse(trimmed);
  } catch {
    return value;
  }
}

function normalize(obj: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) out[k] = tryParseJson(v);
  return out;
}

export function SpanInspector({ span }: Props) {
  const partitioned = useMemo(() => {
    if (!span) return { input: {}, output: {}, rest: {} };
    const attrs = span.attributes ?? {};
    const input = normalize(pick(attrs, (k) => INPUT_KEYS.has(k)));
    const output = normalize(pick(attrs, (k) => OUTPUT_KEYS.has(k)));
    const rest = normalize(
      pick(attrs, (k) => !INPUT_KEYS.has(k) && !OUTPUT_KEYS.has(k))
    );
    return { input, output, rest };
  }, [span]);

  if (!span) {
    return (
      <div className="flex h-full items-center justify-center rounded-md border border-dashed p-12 text-center text-sm text-muted-foreground">
        Select a span to inspect it.
      </div>
    );
  }

  const duration =
    new Date(span.end_time).getTime() - new Date(span.start_time).getTime();

  return (
    <div className="flex flex-col rounded-md border bg-card">
      <div className="flex items-start justify-between gap-3 border-b px-4 py-3">
        <div className="min-w-0">
          <div className="truncate font-mono text-sm">{span.name}</div>
          <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
            <span className="font-mono">{span.span_id.slice(0, 10)}…</span>
            <span>·</span>
            <span>{formatDurationMs(duration)}</span>
          </div>
        </div>
        <TraceStatusBadge status={span.status} />
      </div>
      <Tabs defaultValue="input" className="flex-1">
        <TabsList className="mx-4 mt-3">
          <TabsTrigger value="input">Input</TabsTrigger>
          <TabsTrigger value="output">Output</TabsTrigger>
          <TabsTrigger value="attributes">Attributes</TabsTrigger>
          <TabsTrigger value="events">Events</TabsTrigger>
        </TabsList>
        <TabsContent value="input" className="p-4 pt-3">
          <PaneContent value={partitioned.input} emptyLabel="No input captured." />
        </TabsContent>
        <TabsContent value="output" className="p-4 pt-3">
          <PaneContent value={partitioned.output} emptyLabel="No output captured." />
        </TabsContent>
        <TabsContent value="attributes" className="p-4 pt-3">
          <PaneContent value={partitioned.rest} emptyLabel="No attributes." />
        </TabsContent>
        <TabsContent value="events" className="p-4 pt-3">
          {span.events.length > 0 ? (
            <JsonViewer data={span.events} />
          ) : (
            <p className="text-xs text-muted-foreground">No events emitted.</p>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}

function PaneContent({
  value,
  emptyLabel,
}: {
  value: Record<string, unknown>;
  emptyLabel: string;
}) {
  if (!value || Object.keys(value).length === 0) {
    return <p className="text-xs text-muted-foreground">{emptyLabel}</p>;
  }
  return <JsonViewer data={value} />;
}
