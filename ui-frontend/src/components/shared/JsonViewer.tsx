import { useState } from "react";
import { ChevronDown, ChevronRight, Copy, Check } from "lucide-react";

interface JsonViewerProps {
  data: unknown;
  collapsed?: boolean;
  className?: string;
}

export function JsonViewer({ data, collapsed = false, className = "" }: JsonViewerProps) {
  const [copied, setCopied] = useState(false);

  function handleCopy() {
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className={`relative group rounded-md bg-fa-terminal text-fa-terminal-fg font-mono text-xs overflow-auto ${className}`}>
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded bg-muted/20 hover:bg-muted/40 text-muted-foreground"
        title="Copy JSON"
      >
        {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
      </button>
      <div className="p-3">
        <JsonValue value={data} depth={0} defaultCollapsed={collapsed} />
      </div>
    </div>
  );
}

function JsonValue({
  value,
  depth,
  defaultCollapsed,
}: {
  value: unknown;
  depth: number;
  defaultCollapsed: boolean;
}) {
  if (value === null) return <span className="json-null">null</span>;
  if (value === undefined) return <span className="json-null">undefined</span>;

  if (typeof value === "boolean") {
    return <span className="json-boolean">{value.toString()}</span>;
  }

  if (typeof value === "number") {
    return <span className="json-number">{value}</span>;
  }

  if (typeof value === "string") {
    return <span className="json-string">"{value}"</span>;
  }

  if (Array.isArray(value)) {
    return (
      <CollapsibleJson
        open={"{"}
        close={"]"}
        isEmpty={value.length === 0}
        depth={depth}
        defaultCollapsed={defaultCollapsed && depth > 0}
        count={value.length}
      >
        {value.map((item, i) => (
          <div key={i} className="flex" style={{ paddingLeft: `${(depth + 1) * 16}px` }}>
            <JsonValue value={item} depth={depth + 1} defaultCollapsed={defaultCollapsed} />
            {i < value.length - 1 && <span className="json-bracket">,</span>}
          </div>
        ))}
      </CollapsibleJson>
    );
  }

  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    return (
      <CollapsibleJson
        open={"{"}
        close={"}"}
        isEmpty={entries.length === 0}
        depth={depth}
        defaultCollapsed={defaultCollapsed && depth > 0}
        count={entries.length}
      >
        {entries.map(([key, val], i) => (
          <div key={key} className="flex flex-wrap" style={{ paddingLeft: `${(depth + 1) * 16}px` }}>
            <span className="json-key">"{key}"</span>
            <span className="json-bracket">: </span>
            <JsonValue value={val} depth={depth + 1} defaultCollapsed={defaultCollapsed} />
            {i < entries.length - 1 && <span className="json-bracket">,</span>}
          </div>
        ))}
      </CollapsibleJson>
    );
  }

  return <span>{String(value)}</span>;
}

function CollapsibleJson({
  open,
  close,
  isEmpty,
  children,
  depth,
  defaultCollapsed,
  count,
}: {
  open: string;
  close: string;
  isEmpty: boolean;
  children: React.ReactNode;
  depth: number;
  defaultCollapsed: boolean;
  count: number;
}) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);

  if (isEmpty) {
    return (
      <span className="json-bracket">
        {open === "{" ? "{}" : "[]"}
      </span>
    );
  }

  if (collapsed) {
    return (
      <span>
        <button
          onClick={() => setCollapsed(false)}
          className="inline-flex items-center hover:bg-muted/20 rounded"
        >
          <ChevronRight className="h-3 w-3 text-muted-foreground" />
        </button>
        <span className="json-bracket">{open}</span>
        <span className="text-muted-foreground"> {count} items </span>
        <span className="json-bracket">{close}</span>
      </span>
    );
  }

  return (
    <span>
      <button
        onClick={() => setCollapsed(true)}
        className="inline-flex items-center hover:bg-muted/20 rounded"
      >
        <ChevronDown className="h-3 w-3 text-muted-foreground" />
      </button>
      <span className="json-bracket">{open}</span>
      <div>{children}</div>
      <div style={{ paddingLeft: `${depth * 16}px` }}>
        <span className="json-bracket">{close}</span>
      </div>
    </span>
  );
}
