import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { Wrench, Flag, Square } from "lucide-react";
import type { TopologyNodeData } from "./AgentNode";

type RFFunctionNode = Node<TopologyNodeData, "tool">;

export function FunctionNode({ data }: NodeProps<RFFunctionNode>) {
  const t = data.topology;
  const Icon =
    t.type === "start" ? Flag : t.type === "end" ? Square : Wrench;
  return (
    <div
      className="rounded-md border-2 border-border bg-muted/40 px-3 py-2 shadow-sm hover:border-foreground/40 transition-colors min-w-[140px]"
      data-node-type={t.type}
    >
      <Handle type="target" position={Position.Left} className="!bg-foreground/40" />
      <Handle type="source" position={Position.Right} className="!bg-foreground/40" />
      <div className="flex items-center gap-1.5">
        <Icon className="h-3.5 w-3.5 text-muted-foreground" />
        <div className="font-mono text-xs font-medium truncate flex-1">
          {t.label}
        </div>
      </div>
      {t.tool_name ? (
        <div className="mt-1 font-mono text-[10px] text-muted-foreground truncate">
          {t.tool_name}
        </div>
      ) : null}
    </div>
  );
}
