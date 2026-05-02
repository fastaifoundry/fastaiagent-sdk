/**
 * Custom edges for the workflow topology canvas.
 *
 * Each edge type renders the same Bezier path with type-specific styling:
 *   - Sequential: solid, default color, no label
 *   - Conditional: solid, violet, shows the condition expression
 *   - Handoff: solid, primary color, "handoff" label
 *   - Delegation: dashed, emerald, "delegate" label
 */
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  type Edge,
  type EdgeProps,
} from "@xyflow/react";

export type TopologyEdgeData = {
  type: "sequential" | "conditional" | "handoff" | "delegation";
  label?: string;
  condition?: string;
} & Record<string, unknown>;

type RFTopologyEdge<T extends string> = Edge<TopologyEdgeData, T>;

function EdgeLabelChip({
  x,
  y,
  text,
  className,
}: {
  x: number;
  y: number;
  text: string;
  className: string;
}) {
  return (
    <EdgeLabelRenderer>
      <div
        style={{
          position: "absolute",
          transform: `translate(-50%, -50%) translate(${x}px, ${y}px)`,
          pointerEvents: "all",
        }}
        className={`px-1.5 py-0.5 rounded font-mono text-[10px] ${className}`}
      >
        {text}
      </div>
    </EdgeLabelRenderer>
  );
}

export function SequentialEdge(props: EdgeProps<RFTopologyEdge<"sequential">>) {
  const [edgePath] = getBezierPath(props);
  return <BaseEdge id={props.id} path={edgePath} />;
}

export function ConditionalEdge(props: EdgeProps<RFTopologyEdge<"conditional">>) {
  const [edgePath, labelX, labelY] = getBezierPath(props);
  const cond =
    (props.data as TopologyEdgeData | undefined)?.condition || "if";
  return (
    <>
      <BaseEdge
        id={props.id}
        path={edgePath}
        style={{ stroke: "var(--violet-500, rgb(139 92 246))", strokeWidth: 1.5 }}
      />
      <EdgeLabelChip
        x={labelX}
        y={labelY}
        text={cond}
        className="bg-violet-500/10 text-violet-700 dark:text-violet-300 border border-violet-500/30"
      />
    </>
  );
}

export function HandoffEdge(props: EdgeProps<RFTopologyEdge<"handoff">>) {
  const [edgePath, labelX, labelY] = getBezierPath(props);
  return (
    <>
      <BaseEdge
        id={props.id}
        path={edgePath}
        style={{ stroke: "hsl(var(--primary))", strokeWidth: 1.5 }}
      />
      <EdgeLabelChip
        x={labelX}
        y={labelY}
        text="handoff"
        className="bg-primary/10 text-primary border border-primary/30"
      />
    </>
  );
}

export function DelegationEdge(props: EdgeProps<RFTopologyEdge<"delegation">>) {
  const [edgePath, labelX, labelY] = getBezierPath(props);
  return (
    <>
      <BaseEdge
        id={props.id}
        path={edgePath}
        style={{
          stroke: "rgb(16 185 129)",
          strokeWidth: 1.5,
          strokeDasharray: "5,5",
        }}
      />
      <EdgeLabelChip
        x={labelX}
        y={labelY}
        text="delegate"
        className="bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border border-emerald-500/30"
      />
    </>
  );
}
