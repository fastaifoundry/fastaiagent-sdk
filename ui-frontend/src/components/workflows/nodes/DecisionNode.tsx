import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { GitBranch } from "lucide-react";
import type { TopologyNodeData } from "./AgentNode";

type RFDecisionNode = Node<TopologyNodeData, "condition">;

export function DecisionNode({ data }: NodeProps<RFDecisionNode>) {
  const t = data.topology;
  return (
    <div
      className="relative rounded-md border-2 border-violet-500/70 bg-violet-500/5 px-3 py-2 shadow-sm hover:border-violet-500 transition-colors min-w-[140px]"
      data-node-type="condition"
    >
      <Handle
        type="target"
        position={Position.Left}
        className="!bg-violet-500/60"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!bg-violet-500/60"
      />
      <div className="flex items-center gap-1.5">
        <GitBranch className="h-3.5 w-3.5 text-violet-600 dark:text-violet-400" />
        <div className="font-mono text-xs font-medium truncate flex-1">
          {t.label}
        </div>
      </div>
      <div className="mt-1 text-[10px] text-violet-700 dark:text-violet-400">
        Decision
      </div>
    </div>
  );
}
