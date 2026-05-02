import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { Bot, CircleDot } from "lucide-react";
import type { TopologyNode } from "@/lib/types";

export type TopologyNodeData = {
  topology: TopologyNode;
  isEntrypoint: boolean;
} & Record<string, unknown>;
type RFTopologyNode = Node<TopologyNodeData, "agent">;

export function AgentNode({ data }: NodeProps<RFTopologyNode>) {
  const t = data.topology;
  return (
    <div
      className="rounded-md border-2 border-primary/60 bg-card px-3 py-2 shadow-sm hover:border-primary transition-colors min-w-[160px]"
      data-node-type="agent"
    >
      <Handle type="target" position={Position.Left} className="!bg-primary/60" />
      <Handle type="source" position={Position.Right} className="!bg-primary/60" />
      <div className="flex items-center gap-1.5">
        <Bot className="h-3.5 w-3.5 text-primary" />
        <div className="font-mono text-xs font-medium truncate flex-1">
          {t.label}
        </div>
        {data.isEntrypoint ? (
          <CircleDot
            className="h-3 w-3 text-green-600 dark:text-green-400"
            aria-label="entry point"
          />
        ) : null}
      </div>
      {t.agent_name && t.agent_name !== t.label ? (
        <div className="mt-1 font-mono text-[10px] text-primary truncate">
          {t.agent_name}
        </div>
      ) : null}
      {t.model ? (
        <div className="mt-1 font-mono text-[10px] text-muted-foreground truncate">
          {t.model}
        </div>
      ) : null}
      {(t.tool_count ?? 0) > 0 ? (
        <div className="mt-1 text-[10px] text-muted-foreground">
          {t.tool_count} tool{(t.tool_count ?? 0) === 1 ? "" : "s"}
        </div>
      ) : null}
    </div>
  );
}
