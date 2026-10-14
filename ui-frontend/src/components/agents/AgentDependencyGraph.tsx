/**
 * Structural "what is this agent made of" graph.
 *
 * Renders the response of ``GET /api/agents/:name/dependencies`` as a React
 * Flow canvas: agent at the centre, with tools / knowledge-bases / prompts /
 * guardrails / model / sub-agents radiating out. Reuses ``@xyflow/react`` so
 * the visual language matches the workflow topology view.
 *
 * Sprint 2 / Feature 2.
 */
import { useMemo, useState } from "react";
import {
  Background,
  Controls,
  Handle,
  Position,
  ReactFlow,
  type Edge as RFEdge,
  type Node as RFNode,
  type NodeMouseHandler,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import dagre from "dagre";
import { Link } from "react-router-dom";
import {
  Bot,
  ExternalLink,
  FileText,
  Hammer,
  Cpu,
  Database,
  Shield,
  Users,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { useAgentDependencies } from "@/hooks/use-agents";
import type {
  AgentDepGuardrail,
  AgentDepKB,
  AgentDepNode,
  AgentDepObservedGuardrail,
  AgentDepPrompt,
  AgentDepTool,
  AgentDependencies,
} from "@/lib/types";
import { cn } from "@/lib/utils";
import { formatDurationMs } from "@/lib/format";
import { guardrailEventsHref } from "@/lib/guardrail";

// ---------------------------------------------------------------------------
// Node renderers — small, self-contained. Each takes a typed payload from
// data.payload and renders a labelled chip.
// ---------------------------------------------------------------------------

type NodeKind = "agent" | "tool" | "kb" | "prompt" | "model" | "guardrail" | "worker";

interface NodeBaseData {
  kind: NodeKind;
  title: string;
  subtitle?: string;
  badge?: string;
  warn?: boolean;
  /** Drawn dashed: inferred from runs rather than declared (observed rules). */
  dashed?: boolean;
  payload?: unknown;
}

function NodeShell({
  icon: Icon,
  data,
  className,
}: {
  icon: React.ComponentType<{ className?: string }>;
  data: NodeBaseData;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "rounded-md border bg-card px-3 py-2 shadow-sm transition-colors",
        "min-w-[160px] max-w-[220px]",
        data.warn ? "border-fa-warning" : "hover:border-primary/50",
        data.dashed && "border-dashed",
        className,
      )}
      data-node-kind={data.kind}
      data-observed={data.dashed ? "true" : undefined}
    >
      <Handle type="target" position={Position.Top} className="!opacity-0" />
      <div className="flex items-center gap-1.5">
        <Icon className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          {data.kind}
        </span>
        {data.badge && (
          <span
            className={cn(
              "ml-auto rounded-sm px-1.5 py-0.5 text-[9px] font-mono uppercase",
              data.warn
                ? "bg-fa-warning/10 text-fa-warning"
                : "bg-muted text-muted-foreground",
              data.dashed && "border border-dashed border-muted-foreground/50",
            )}
          >
            {data.badge}
          </span>
        )}
      </div>
      <div className="mt-0.5 truncate font-mono text-xs" title={data.title}>
        {data.title}
      </div>
      {data.subtitle && (
        <div
          className="truncate text-[10px] text-muted-foreground"
          title={data.subtitle}
        >
          {data.subtitle}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} className="!opacity-0" />
    </div>
  );
}

function AgentCenterNode({ data }: NodeProps) {
  const d = data as unknown as NodeBaseData;
  return (
    <NodeShell
      icon={Bot}
      data={d}
      className="border-primary/60 bg-primary/5"
    />
  );
}

function ToolDepNode({ data }: NodeProps) {
  const d = data as unknown as NodeBaseData;
  return <NodeShell icon={Hammer} data={d} />;
}

function KbDepNode({ data }: NodeProps) {
  const d = data as unknown as NodeBaseData;
  return <NodeShell icon={Database} data={d} />;
}

function PromptDepNode({ data }: NodeProps) {
  const d = data as unknown as NodeBaseData;
  return <NodeShell icon={FileText} data={d} />;
}

function ModelDepNode({ data }: NodeProps) {
  const d = data as unknown as NodeBaseData;
  return <NodeShell icon={Cpu} data={d} />;
}

function GuardrailDepNode({ data }: NodeProps) {
  const d = data as unknown as NodeBaseData;
  return <NodeShell icon={Shield} data={d} />;
}

function WorkerDepNode({ data }: NodeProps) {
  const d = data as unknown as NodeBaseData;
  return <NodeShell icon={Users} data={d} className="border-primary/30" />;
}

const NODE_TYPES = {
  agent: AgentCenterNode,
  tool: ToolDepNode,
  kb: KbDepNode,
  prompt: PromptDepNode,
  model: ModelDepNode,
  guardrail: GuardrailDepNode,
  worker: WorkerDepNode,
} as const;

// ---------------------------------------------------------------------------
// Layout — dagre, top-down. Sub-agents render as a second tier under the
// centre node; their dependency clusters fan out from each worker.
// ---------------------------------------------------------------------------

const NODE_W = 200;
const NODE_H = 60;

interface BuiltGraph {
  nodes: RFNode[];
  edges: RFEdge[];
}

/** A guardrail node's payload: the rule plus where it applies. */
interface GuardrailNodePayload {
  agent: string;
  name: string | null;
  guardrail_type: string | null;
  position: string | null;
  /** Ran for the agent without being declared on it (e.g. plane-distributed). */
  observed: boolean;
  declared?: AgentDepGuardrail;
  seen?: AgentDepObservedGuardrail;
}

type AddNode = (
  id: string,
  type: NodeKind,
  title: string,
  subtitle?: string,
  badge?: string,
  warn?: boolean,
  payload?: unknown,
  dashed?: boolean,
) => void;

/**
 * One node per rule. The id carries position and index: the same rule name
 * declared at input and at output (two ``no_pii``) is two rules, and
 * collapsing them by name hid one.
 */
function addGuardrailNodes(
  deps: AgentDependencies,
  parentId: string,
  addNode: AddNode,
  addEdge: (from: string, to: string) => void,
) {
  const agent = deps.agent.name;
  const declared = deps.guardrails;
  declared.forEach((gr, index) => {
    const id = `guardrail:${agent}:${gr.position}:${gr.name}:${index}`;
    const payload: GuardrailNodePayload = {
      agent,
      name: gr.name,
      guardrail_type: gr.guardrail_type,
      position: gr.position,
      observed: false,
      declared: gr,
    };
    addNode(
      id,
      "guardrail",
      gr.name ?? "guardrail",
      [gr.guardrail_type, gr.blocking === false ? "observe-only" : gr.action]
        .filter(Boolean)
        .join(" · ") || undefined,
      gr.position ?? undefined,
      false,
      payload,
    );
    addEdge(parentId, id);
  });
  (deps.observed_guardrails ?? []).forEach((gr, i) => {
    const index = declared.length + i;
    const id = `guardrail:${agent}:${gr.position}:${gr.name}:${index}`;
    const payload: GuardrailNodePayload = {
      agent,
      name: gr.name,
      guardrail_type: gr.guardrail_type,
      position: gr.position,
      observed: true,
      seen: gr,
    };
    addNode(
      id,
      "guardrail",
      gr.name,
      [gr.guardrail_type, gr.position, gr.origin].filter(Boolean).join(" · ") || undefined,
      "observed",
      false,
      payload,
      true,
    );
    addEdge(parentId, id);
  });
}

function buildGraph(deps: AgentDependencies): BuiltGraph {
  const nodes: RFNode[] = [];
  const edges: RFEdge[] = [];
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({
    rankdir: "TB",
    nodesep: 30,
    ranksep: 60,
    marginx: 20,
    marginy: 20,
  });

  const addNode: AddNode = (
    id,
    type,
    title,
    subtitle,
    badge,
    warn,
    payload,
    dashed,
  ) => {
    g.setNode(id, { width: NODE_W, height: NODE_H });
    nodes.push({
      id,
      type,
      position: { x: 0, y: 0 },
      data: {
        kind: type,
        title,
        subtitle,
        badge,
        warn,
        dashed,
        payload,
      } as unknown as Record<string, unknown>,
    });
  };

  const addEdge = (from: string, to: string) => {
    g.setEdge(from, to);
    edges.push({
      id: `${from}->${to}`,
      source: from,
      target: to,
      type: "default",
    });
  };

  // Centre — the agent itself.
  const centerId = `agent:${deps.agent.name}`;
  addNode(
    centerId,
    deps.agent.type === "supervisor" ? "agent" : "agent",
    deps.agent.name,
    deps.agent.type,
    deps.unresolved ? "unresolved" : undefined,
    deps.unresolved,
    deps.agent,
  );

  // Model node (always shown if known).
  if (deps.model.model) {
    const modelId = `model:${deps.model.provider}/${deps.model.model}`;
    addNode(
      modelId,
      "model",
      deps.model.model,
      deps.model.provider ?? undefined,
      undefined,
      false,
      deps.model,
    );
    addEdge(centerId, modelId);
  }

  // Tools.
  for (const t of deps.tools) {
    const id = `tool:${deps.agent.name}:${t.name}`;
    const subtitle =
      t.calls > 0
        ? `${t.calls} call${t.calls === 1 ? "" : "s"} · ${formatDurationMs(t.avg_latency_ms)}`
        : t.origin;
    addNode(
      id,
      "tool",
      t.name,
      subtitle,
      t.registered ? t.origin : "unregistered",
      !t.registered,
      t,
    );
    addEdge(centerId, id);
  }

  // Knowledge bases.
  for (const kb of deps.knowledge_bases) {
    const id = `kb:${deps.agent.name}:${kb.name}`;
    const subtitle =
      kb.chunks != null ? `${kb.chunks} chunks · ${kb.backend}` : kb.backend;
    addNode(
      id,
      "kb",
      kb.name,
      subtitle,
      kb.unresolved ? "unresolved" : undefined,
      kb.unresolved,
      kb,
    );
    addEdge(centerId, id);
  }

  // Prompts.
  for (const p of deps.prompts) {
    const id = `prompt:${deps.agent.name}:${p.name}`;
    const subtitle =
      p.variables.length > 0
        ? `vars: ${p.variables.map((v) => `{{${v}}}`).join(" ")}`
        : "no variables";
    addNode(
      id,
      "prompt",
      p.name,
      subtitle,
      p.version ?? undefined,
      false,
      p,
    );
    addEdge(centerId, id);
  }

  // Guardrails — declared, then observed.
  addGuardrailNodes(deps, centerId, addNode, addEdge);

  // Sub-agents (Supervisor workers). Each sub-agent gets its own subtree.
  for (const sub of deps.sub_agents ?? []) {
    const subId = `worker:${sub.agent.name}`;
    addNode(
      subId,
      "worker",
      sub.agent.name,
      sub.role ? `role: ${sub.role}` : undefined,
      sub.agent.model ?? undefined,
      false,
      sub,
    );
    addEdge(centerId, subId);

    if (sub.model.model) {
      const subModelId = `model:${sub.agent.name}:${sub.model.model}`;
      addNode(
        subModelId,
        "model",
        sub.model.model,
        sub.model.provider ?? undefined,
        undefined,
        false,
        sub.model,
      );
      addEdge(subId, subModelId);
    }
    for (const t of sub.tools) {
      const id = `tool:${sub.agent.name}:${t.name}`;
      addNode(
        id,
        "tool",
        t.name,
        t.origin,
        t.registered ? t.origin : "unregistered",
        !t.registered,
        t,
      );
      addEdge(subId, id);
    }
    for (const kb of sub.knowledge_bases) {
      const id = `kb:${sub.agent.name}:${kb.name}`;
      addNode(id, "kb", kb.name, kb.backend, undefined, false, kb);
      addEdge(subId, id);
    }
    for (const p of sub.prompts) {
      const id = `prompt:${sub.agent.name}:${p.name}`;
      addNode(id, "prompt", p.name, undefined, undefined, false, p);
      addEdge(subId, id);
    }
    addGuardrailNodes(sub, subId, addNode, addEdge);
  }

  // Swarm peers — flat siblings of the centre, with handoff edges drawn
  // between them. Their dependency clusters aren't expanded (single-agent
  // detail page is the better surface for that — click through).
  for (const peer of deps.peers ?? []) {
    const peerId = `peer:${peer.name}`;
    addNode(
      peerId,
      "worker",
      peer.name,
      peer.model ?? undefined,
      "peer",
      false,
      peer,
    );
    // No direct edge from the centre — handoff edges below carry the
    // semantic.
  }
  for (const h of deps.handoffs ?? []) {
    const fromId = h.from === deps.agent.name ? centerId : `peer:${h.from}`;
    const toId = h.to === deps.agent.name ? centerId : `peer:${h.to}`;
    g.setEdge(fromId, toId);
    edges.push({
      id: `handoff:${h.from}->${h.to}`,
      source: fromId,
      target: toId,
      type: "default",
      animated: true,
    });
  }

  dagre.layout(g);

  for (const n of nodes) {
    const pos = g.node(n.id);
    n.position = {
      x: (pos?.x ?? 0) - NODE_W / 2,
      y: (pos?.y ?? 0) - NODE_H / 2,
    };
  }

  return { nodes, edges };
}

// ---------------------------------------------------------------------------
// Detail panel — opens on node click with type-aware contents and an
// "Open" link to the dependency's own page where applicable.
// ---------------------------------------------------------------------------

function DetailPanel({
  open,
  onOpenChange,
  data,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  data: NodeBaseData | null;
}) {
  if (!data) return null;
  const linkFor = (kind: NodeKind, payload: unknown): string | null => {
    if (kind === "kb" && (payload as AgentDepKB)?.name)
      return `/kb/${encodeURIComponent((payload as AgentDepKB).name)}`;
    if (kind === "prompt" && (payload as AgentDepPrompt)?.name)
      return `/prompts/${encodeURIComponent((payload as AgentDepPrompt).name)}`;
    if (kind === "worker" && (payload as AgentDependencies)?.agent)
      return `/agents/${encodeURIComponent(
        (payload as AgentDependencies).agent.name,
      )}`;
    if (kind === "agent" && (payload as AgentDepNode)?.name)
      return `/agents/${encodeURIComponent((payload as AgentDepNode).name)}`;
    if (kind === "guardrail" && (payload as GuardrailNodePayload)?.name)
      return guardrailEventsHref(
        (payload as GuardrailNodePayload).agent,
        (payload as GuardrailNodePayload).name,
        (payload as GuardrailNodePayload).position,
      );
    return null;
  };

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent>
        <SheetHeader>
          <SheetTitle>
            <span className="font-mono">{data.title}</span>
          </SheetTitle>
          <SheetDescription>
            {data.kind} · {data.subtitle ?? "no metadata"}
          </SheetDescription>
        </SheetHeader>

        <div className="px-4 py-3 space-y-3 text-sm">
          {data.kind === "tool" && (
            <ToolDetail tool={data.payload as AgentDepTool} />
          )}
          {data.kind === "kb" && <KbDetail kb={data.payload as AgentDepKB} />}
          {data.kind === "prompt" && (
            <PromptDetail prompt={data.payload as AgentDepPrompt} />
          )}
          {data.kind === "guardrail" && (
            <GuardrailDetail guardrail={data.payload as GuardrailNodePayload} />
          )}
          {data.kind === "worker" && (
            <WorkerDetail worker={data.payload as AgentDependencies} />
          )}
          {data.kind === "agent" && (
            <AgentDetail agent={data.payload as AgentDepNode} />
          )}
          {data.kind === "model" && (
            <pre className="rounded-md bg-muted/30 p-2 font-mono text-xs">
              {JSON.stringify(data.payload, null, 2)}
            </pre>
          )}
        </div>

        {(() => {
          const href = linkFor(data.kind, data.payload);
          if (!href) return null;
          return (
            <div className="px-4 pb-4">
              <Link to={href}>
                <Button variant="outline" size="sm">
                  <ExternalLink className="mr-1.5 h-3.5 w-3.5" />
                  Open
                </Button>
              </Link>
            </div>
          );
        })()}
      </SheetContent>
    </Sheet>
  );
}

function ToolDetail({ tool }: { tool: AgentDepTool }) {
  return (
    <ul className="space-y-1.5">
      <li>
        <span className="text-muted-foreground">origin:</span>{" "}
        <span className="font-mono">{tool.origin}</span>
      </li>
      <li>
        <span className="text-muted-foreground">registered:</span>{" "}
        <span
          className={cn(
            "font-mono",
            tool.registered ? "text-fa-success" : "text-fa-warning",
          )}
        >
          {String(tool.registered)}
        </span>
      </li>
      <li>
        <span className="text-muted-foreground">calls:</span>{" "}
        <span className="font-mono">{tool.calls}</span>
      </li>
      <li>
        <span className="text-muted-foreground">success rate:</span>{" "}
        <span className="font-mono">
          {tool.calls > 0 ? `${Math.round(tool.success_rate * 100)}%` : "—"}
        </span>
      </li>
      <li>
        <span className="text-muted-foreground">avg latency:</span>{" "}
        <span className="font-mono">
          {tool.calls > 0 ? formatDurationMs(tool.avg_latency_ms) : "—"}
        </span>
      </li>
    </ul>
  );
}

function KbDetail({ kb }: { kb: AgentDepKB }) {
  return (
    <ul className="space-y-1.5">
      <li>
        <span className="text-muted-foreground">backend:</span>{" "}
        <span className="font-mono">{kb.backend}</span>
      </li>
      <li>
        <span className="text-muted-foreground">chunks:</span>{" "}
        <span className="font-mono">{kb.chunks ?? "—"}</span>
      </li>
      {kb.unresolved && (
        <li className="text-fa-warning">
          KB metadata couldn't be loaded — file may not exist on disk.
        </li>
      )}
    </ul>
  );
}

function PromptDetail({ prompt }: { prompt: AgentDepPrompt }) {
  return (
    <div className="space-y-2">
      <ul className="space-y-1.5">
        <li>
          <span className="text-muted-foreground">version:</span>{" "}
          <span className="font-mono">{prompt.version ?? "—"}</span>
        </li>
        <li>
          <span className="text-muted-foreground">variables:</span>{" "}
          <span className="font-mono">
            {prompt.variables.length > 0
              ? prompt.variables.map((v) => `{{${v}}}`).join(" ")
              : "—"}
          </span>
        </li>
      </ul>
      {prompt.preview && (
        <pre className="rounded-md border bg-muted/30 p-2 font-mono text-xs whitespace-pre-wrap">
          {prompt.preview}
        </pre>
      )}
    </div>
  );
}

function GuardrailDetail({ guardrail }: { guardrail: GuardrailNodePayload }) {
  const d = guardrail.declared;
  const seen = guardrail.seen;
  const field = (label: string, value: unknown) =>
    value == null ? null : (
      <li>
        <span className="text-muted-foreground">{label}:</span>{" "}
        <span className="font-mono">{String(value)}</span>
      </li>
    );
  return (
    <ul className="space-y-1.5">
      {field("type", guardrail.guardrail_type ?? "—")}
      {field("position", guardrail.position ?? "—")}
      {d && (
        <>
          {field("action", d.action)}
          {d.blocking != null && field("mode", d.blocking ? "blocking" : "observe-only")}
          {field("on_error", d.on_error)}
          {field("severity", d.severity)}
          {d.floor && field("floor", "yes")}
        </>
      )}
      {seen && (
        <>
          <li className="text-xs text-muted-foreground">
            Observed: this rule ran for the agent but is not declared on it
            (e.g. distributed by a control plane).
          </li>
          {field("origin", seen.origin)}
          {field("checks", seen.checks)}
          {field("last seen", seen.last_seen)}
        </>
      )}
    </ul>
  );
}

function AgentDetail({ agent }: { agent: AgentDepNode }) {
  return (
    <ul className="space-y-1.5">
      <li>
        <span className="text-muted-foreground">type:</span>{" "}
        <span className="font-mono">{agent.type}</span>
      </li>
      <li>
        <span className="text-muted-foreground">model:</span>{" "}
        <span className="font-mono">
          {agent.provider ?? "?"}/{agent.model ?? "?"}
        </span>
      </li>
    </ul>
  );
}

function WorkerDetail({ worker }: { worker: AgentDependencies }) {
  return (
    <ul className="space-y-1.5">
      <li>
        <span className="text-muted-foreground">role:</span>{" "}
        <span className="font-mono">{worker.role ?? "—"}</span>
      </li>
      <li>
        <span className="text-muted-foreground">tools:</span>{" "}
        <span className="font-mono">{worker.tools.length}</span>
      </li>
      <li>
        <span className="text-muted-foreground">model:</span>{" "}
        <span className="font-mono">
          {worker.model.provider ?? "?"}/{worker.model.model ?? "?"}
        </span>
      </li>
    </ul>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

interface Props {
  agentName: string;
  height?: number;
}

export function AgentDependencyGraph({ agentName, height = 480 }: Props) {
  const dependencies = useAgentDependencies(agentName);

  const [open, setOpen] = useState(false);
  const [active, setActive] = useState<NodeBaseData | null>(null);

  const graph = useMemo<BuiltGraph>(() => {
    if (!dependencies.data) return { nodes: [], edges: [] };
    return buildGraph(dependencies.data);
  }, [dependencies.data]);

  const handleNodeClick: NodeMouseHandler = (_e, node) => {
    setActive(node.data as unknown as NodeBaseData);
    setOpen(true);
  };

  if (dependencies.isLoading) return <TableSkeleton rows={4} />;
  if (dependencies.error || !dependencies.data) {
    return (
      <EmptyState
        title="No dependency graph yet"
        description="Register the agent's runner via build_app(runners=[…]) so the UI can introspect it."
      />
    );
  }

  return (
    <div className="space-y-3">
      {dependencies.data.unresolved && (
        <p className="rounded-md border border-fa-warning bg-fa-warning/5 px-3 py-2 text-xs text-fa-warning">
          Showing reduced data reconstructed from past traces — register this
          agent's runner via{" "}
          <code className="font-mono">build_app(runners=[…])</code> to see the
          full structural view.
        </p>
      )}
      <div
        data-testid="agent-dependency-graph"
        className="rounded-md border bg-card"
        style={{ height }}
      >
        <ReactFlow
          nodes={graph.nodes}
          edges={graph.edges}
          nodeTypes={NODE_TYPES}
          onNodeClick={handleNodeClick}
          fitView
          nodesConnectable={false}
          nodesDraggable={false}
          edgesFocusable={false}
          panOnScroll
          zoomOnScroll
          zoomOnPinch
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={16} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
      <DetailPanel open={open} onOpenChange={setOpen} data={active} />
    </div>
  );
}
