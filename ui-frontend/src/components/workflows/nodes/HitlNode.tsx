import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { Pause } from "lucide-react";
import type { TopologyNodeData } from "./AgentNode";

type RFHitlNode = Node<TopologyNodeData, "hitl">;

export function HitlNode({ data }: NodeProps<RFHitlNode>) {
  const t = data.topology;
  return (
    <div
      className="rounded-md border-2 border-amber-500/70 bg-amber-500/5 px-3 py-2 shadow-sm hover:border-amber-500 transition-colors min-w-[160px]"
      data-node-type="hitl"
    >
      <Handle type="target" position={Position.Left} className="!bg-amber-500/60" />
      <Handle type="source" position={Position.Right} className="!bg-amber-500/60" />
      <div className="flex items-center gap-1.5">
        <Pause className="h-3.5 w-3.5 text-amber-600 dark:text-amber-400" />
        <div className="font-mono text-xs font-medium truncate flex-1">
          {t.label}
        </div>
      </div>
      <div className="mt-1 text-[10px] text-amber-700 dark:text-amber-400">
        Approval gate
      </div>
    </div>
  );
}
