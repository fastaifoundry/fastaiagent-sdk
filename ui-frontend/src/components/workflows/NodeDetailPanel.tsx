import { X } from "lucide-react";
import type { TopologyNode, TopologyTool } from "@/lib/types";
import { Button } from "@/components/ui/button";

interface Props {
  node: TopologyNode | null;
  tools: TopologyTool[];
  onClose: () => void;
}

export function NodeDetailPanel({ node, tools, onClose }: Props) {
  if (!node) return null;
  const ownTools = tools.filter((t) => t.owner === node.id);
  return (
    <div
      className="absolute right-0 top-0 z-10 h-full w-72 border-l border-border bg-card p-4 shadow-lg overflow-auto"
      role="complementary"
      aria-label={`Details for node ${node.id}`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="font-mono text-sm font-semibold truncate">
            {node.label}
          </div>
          <div className="font-mono text-[10px] text-muted-foreground uppercase tracking-widest mt-0.5">
            {node.type}
          </div>
        </div>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Close details"
          onClick={onClose}
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      <div className="mt-4 space-y-3 text-xs">
        {node.agent_name ? (
          <Field label="Agent">
            <span className="font-mono">{node.agent_name}</span>
          </Field>
        ) : null}
        {node.model ? (
          <Field label="Model">
            <span className="font-mono">{node.model}</span>
          </Field>
        ) : null}
        {node.provider ? (
          <Field label="Provider">
            <span className="font-mono">{node.provider}</span>
          </Field>
        ) : null}
        {node.tool_name ? (
          <Field label="Tool">
            <span className="font-mono">{node.tool_name}</span>
          </Field>
        ) : null}
        {node.description ? (
          <Field label="Description">
            <span>{node.description}</span>
          </Field>
        ) : null}
      </div>

      {ownTools.length > 0 ? (
        <div className="mt-4">
          <div className="font-mono text-[10px] text-muted-foreground uppercase tracking-widest">
            Tools ({ownTools.length})
          </div>
          <ul className="mt-2 space-y-1">
            {ownTools.map((t) => (
              <li
                key={`${t.owner}:${t.name}`}
                className="font-mono text-xs px-2 py-1 rounded bg-muted/40 truncate"
              >
                {t.name}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="font-mono text-[10px] text-muted-foreground uppercase tracking-widest">
        {label}
      </div>
      <div className="mt-0.5">{children}</div>
    </div>
  );
}
