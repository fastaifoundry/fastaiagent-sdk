import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { Crown, CircleDot } from "lucide-react";
import type { TopologyNodeData } from "./AgentNode";

type RFSupervisorNode = Node<TopologyNodeData, "supervisor">;

export function SupervisorNode({ data }: NodeProps<RFSupervisorNode>) {
  const t = data.topology;
  return (
    <div
      className="rounded-md border-2 border-emerald-500/70 bg-emerald-500/5 px-3 py-2 shadow-sm hover:border-emerald-500 transition-colors min-w-[180px]"
      data-node-type="supervisor"
    >
      <Handle
        type="target"
        position={Position.Left}
        className="!bg-emerald-500/60"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!bg-emerald-500/60"
      />
      <div className="flex items-center gap-1.5">
        <Crown className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
        <div className="font-mono text-xs font-medium truncate flex-1">
          {t.label}
        </div>
        {data.isEntrypoint ? (
          <CircleDot className="h-3 w-3 text-green-600 dark:text-green-400" />
        ) : null}
      </div>
      {t.model ? (
        <div className="mt-1 font-mono text-[10px] text-muted-foreground truncate">
          {t.model}
        </div>
      ) : null}
    </div>
  );
}
