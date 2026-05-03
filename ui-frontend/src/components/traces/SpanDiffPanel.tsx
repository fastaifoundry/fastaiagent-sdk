import ReactDiffViewer, { DiffMethod } from "react-diff-viewer-continued";
import type { SpanRow } from "@/lib/types";

interface Props {
  spanA: SpanRow | null;
  spanB: SpanRow | null;
}

function stringify(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

/**
 * Pull a span's "input" / "output" / "attributes" projections out of the
 * SDK's flat ``attributes`` JSON. The exact keys differ by span type
 * (LLM vs tool vs retrieval), so look at all the common ones.
 */
function projectInput(span: SpanRow | null): unknown {
  if (!span) return null;
  const a = span.attributes ?? {};
  return (
    a["gen_ai.prompt"] ??
    a["fastaiagent.gen_ai.prompt"] ??
    a["tool.input"] ??
    a["input"] ??
    null
  );
}

function projectOutput(span: SpanRow | null): unknown {
  if (!span) return null;
  const a = span.attributes ?? {};
  return (
    a["gen_ai.response.text"] ??
    a["gen_ai.completion"] ??
    a["fastaiagent.gen_ai.response.text"] ??
    a["tool.output"] ??
    a["output"] ??
    null
  );
}

function projectAttributes(span: SpanRow | null): Record<string, unknown> {
  if (!span) return {};
  // Hide the bulky payload keys (already shown in their own diff blocks)
  // so the attributes diff focuses on metadata.
  const skip = new Set([
    "gen_ai.prompt",
    "fastaiagent.gen_ai.prompt",
    "gen_ai.response.text",
    "gen_ai.completion",
    "fastaiagent.gen_ai.response.text",
    "tool.input",
    "tool.output",
    "input",
    "output",
  ]);
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(span.attributes ?? {})) {
    if (!skip.has(k)) out[k] = v;
  }
  return out;
}

function DiffBlock({
  label,
  oldText,
  newText,
  empty,
}: {
  label: string;
  oldText: string;
  newText: string;
  empty?: string;
}) {
  if (!oldText && !newText) {
    return (
      <div className="rounded-md border bg-background">
        <div className="border-b px-3 py-1.5 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
          {label}
        </div>
        <p className="px-3 py-2 text-[11px] italic text-muted-foreground">
          {empty ?? "Both spans are empty here."}
        </p>
      </div>
    );
  }
  return (
    <div className="rounded-md border bg-background">
      <div className="border-b px-3 py-1.5 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        {label}
      </div>
      <div className="overflow-x-auto text-xs">
        <ReactDiffViewer
          oldValue={oldText}
          newValue={newText}
          splitView
          compareMethod={DiffMethod.WORDS}
          hideLineNumbers={false}
          useDarkTheme={
            typeof document !== "undefined" &&
            document.documentElement.classList.contains("dark")
          }
          styles={{
            variables: {
              dark: {
                diffViewerBackground: "transparent",
                gutterBackground: "transparent",
              },
              light: {
                diffViewerBackground: "transparent",
                gutterBackground: "transparent",
              },
            },
            contentText: { fontSize: 11 },
            gutter: { padding: "0 6px" },
          }}
        />
      </div>
    </div>
  );
}

/**
 * Renders three diff blocks (input / output / attributes) for one row of
 * the alignment table. Either span can be null when a span is "new in A"
 * or "new in B" — the block then shows the side that has content versus
 * an empty string.
 */
export function SpanDiffPanel({ spanA, spanB }: Props) {
  return (
    <div className="space-y-3">
      <DiffBlock
        label="Input"
        oldText={stringify(projectInput(spanA))}
        newText={stringify(projectInput(spanB))}
      />
      <DiffBlock
        label="Output"
        oldText={stringify(projectOutput(spanA))}
        newText={stringify(projectOutput(spanB))}
      />
      <DiffBlock
        label="Attributes"
        oldText={stringify(projectAttributes(spanA))}
        newText={stringify(projectAttributes(spanB))}
        empty="Both spans report identical metadata."
      />
    </div>
  );
}
