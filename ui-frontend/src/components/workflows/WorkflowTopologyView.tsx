import { useEffect, useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  type Edge as RFEdge,
  type Node as RFNode,
  type NodeMouseHandler,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import dagre from "dagre";
import { useWorkflowTopology } from "@/hooks/use-workflow-topology";
import type {
  TopologyEdge,
  TopologyNode,
  TopologyEdgeType,
  TopologyNodeType,
} from "@/lib/types";
import { AgentNode } from "./nodes/AgentNode";
import { HitlNode } from "./nodes/HitlNode";
import { FunctionNode } from "./nodes/FunctionNode";
import { DecisionNode } from "./nodes/DecisionNode";
import { SupervisorNode } from "./nodes/SupervisorNode";
import {
  ConditionalEdge,
  DelegationEdge,
  HandoffEdge,
  SequentialEdge,
} from "./edges/StyledEdges";
import { NodeDetailPanel } from "./NodeDetailPanel";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Loader2 } from "lucide-react";

const NODE_TYPES = {
  agent: AgentNode,
  hitl: HitlNode,
  tool: FunctionNode,
  start: FunctionNode,
  end: FunctionNode,
  transformer: FunctionNode,
  parallel: FunctionNode,
  condition: DecisionNode,
  supervisor: SupervisorNode,
} as const;

const EDGE_TYPES = {
  sequential: SequentialEdge,
  conditional: ConditionalEdge,
  handoff: HandoffEdge,
  delegation: DelegationEdge,
} as const;

const NODE_W = 200;
const NODE_H = 64;

type Layout = "horizontal" | "vertical";

const LAYOUT_STORAGE_KEY = "fa.workflow.layout";

function readStoredLayout(): Layout {
  if (typeof window === "undefined") return "horizontal";
  const v = window.localStorage.getItem(LAYOUT_STORAGE_KEY);
  return v === "vertical" ? "vertical" : "horizontal";
}

function laidOut(
  nodes: TopologyNode[],
  edges: TopologyEdge[],
  entrypoint: string | null,
  layout: Layout
): { rfNodes: RFNode[]; rfEdges: RFEdge[] } {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({
    rankdir: layout === "horizontal" ? "LR" : "TB",
    nodesep: 40,
    ranksep: 60,
    marginx: 20,
    marginy: 20,
  });

  for (const n of nodes) g.setNode(n.id, { width: NODE_W, height: NODE_H });
  for (const e of edges) g.setEdge(e.from, e.to);
  dagre.layout(g);

  const rfNodes: RFNode[] = nodes.map((n) => {
    const pos = g.node(n.id);
    return {
      id: n.id,
      type: nodeTypeFor(n.type),
      position: { x: (pos?.x ?? 0) - NODE_W / 2, y: (pos?.y ?? 0) - NODE_H / 2 },
      data: { topology: n, isEntrypoint: n.id === entrypoint },
    };
  });

  const rfEdges: RFEdge[] = edges.map((e, idx) => ({
    id: `${e.from}->${e.to}#${idx}`,
    source: e.from,
    target: e.to,
    type: edgeTypeFor(e.type),
    data: { type: e.type, condition: e.condition, label: e.label },
  }));

  return { rfNodes, rfEdges };
}

function nodeTypeFor(t: TopologyNodeType): keyof typeof NODE_TYPES {
  if (t in NODE_TYPES) return t as keyof typeof NODE_TYPES;
  return "tool";
}

function edgeTypeFor(t: TopologyEdgeType): keyof typeof EDGE_TYPES {
  return t in EDGE_TYPES ? (t as keyof typeof EDGE_TYPES) : "sequential";
}

interface Props {
  runnerType: string;
  name: string;
  /** Compact preview mode for embedding in another page (no controls). */
  compact?: boolean;
  /** Visible height in pixels. */
  height?: number;
}

export function WorkflowTopologyView({
  runnerType,
  name,
  compact = false,
  height = 480,
}: Props) {
  const { data, isLoading, error } = useWorkflowTopology(runnerType, name);
  const [layout, setLayout] = useState<Layout>(() => readStoredLayout());
  const [selected, setSelected] = useState<TopologyNode | null>(null);

  useEffect(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(LAYOUT_STORAGE_KEY, layout);
    }
  }, [layout]);

  const computed = useMemo(() => {
    if (!data) return { rfNodes: [], rfEdges: [] };
    return laidOut(data.nodes, data.edges, data.entrypoint, layout);
  }, [data, layout]);

  const onNodeClick: NodeMouseHandler = (_event, node) => {
    if (compact) return;
    const top = (node.data as { topology?: TopologyNode })?.topology;
    if (top) setSelected(top);
  };

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-10 text-sm text-muted-foreground">
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          Loading topology…
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return <NotRegisteredCallout runnerType={runnerType} name={name} />;
  }

  if (!data) return null;

  return (
    <div className="relative">
      {!compact ? (
        <div className="mb-2 flex items-center gap-2">
          <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-widest">
            Layout
          </span>
          <Button
            size="sm"
            variant={layout === "horizontal" ? "default" : "outline"}
            onClick={() => setLayout("horizontal")}
          >
            Horizontal
          </Button>
          <Button
            size="sm"
            variant={layout === "vertical" ? "default" : "outline"}
            onClick={() => setLayout("vertical")}
          >
            Vertical
          </Button>
        </div>
      ) : null}
      <div
        className="relative rounded-md border bg-card overflow-hidden"
        style={{ height }}
        data-testid="workflow-topology"
      >
        <ReactFlow
          nodes={computed.rfNodes}
          edges={computed.rfEdges}
          nodeTypes={NODE_TYPES as Record<string, React.ComponentType<any>>}
          edgeTypes={EDGE_TYPES as Record<string, React.ComponentType<any>>}
          onNodeClick={onNodeClick}
          fitView
          fitViewOptions={{
            // Generous padding so the graph doesn't crowd the canvas
            // edges. Clamp the max zoom so small graphs (3–5 nodes)
            // don't blow up to fill the whole height.
            padding: 0.3,
            maxZoom: compact ? 1 : 0.85,
            minZoom: 0.2,
          }}
          minZoom={0.15}
          maxZoom={2}
          panOnScroll={!compact}
          nodesDraggable={!compact}
          nodesConnectable={false}
          edgesFocusable={false}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={16} />
          {!compact ? <Controls position="bottom-right" showInteractive={false} /> : null}
        </ReactFlow>
        {!compact ? (
          <NodeDetailPanel
            node={selected}
            tools={data.tools}
            onClose={() => setSelected(null)}
          />
        ) : null}
      </div>
    </div>
  );
}

function NotRegisteredCallout({
  runnerType,
  name,
}: {
  runnerType: string;
  name: string;
}) {
  return (
    <Card className="border-dashed">
      <CardContent className="py-6">
        <p className="text-sm font-medium">No topology available</p>
        <p className="mt-1 text-xs text-muted-foreground">
          The {runnerType} <span className="font-mono">{name}</span> is not
          registered with the local UI server. Pass it to{" "}
          <span className="font-mono">build_app(runners=[…])</span> to enable
          topology rendering.
        </p>
        <pre className="mt-3 rounded bg-muted/50 p-3 text-[11px] font-mono overflow-auto">
          {`from fastaiagent.ui.server import build_app

app = build_app(runners=[my_${runnerType}])`}
        </pre>
      </CardContent>
    </Card>
  );
}

export default WorkflowTopologyView;
