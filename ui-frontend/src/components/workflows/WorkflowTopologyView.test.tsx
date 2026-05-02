/**
 * Component tests for WorkflowTopologyView.
 *
 * jsdom doesn't paint, so React Flow positions nodes outside the viewport
 * and skips rendering them by default. We mark the parent height + width
 * inline so RF computes a positive viewport, and we assert on data-attrs
 * (data-node-type) rather than visual layout.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { WorkflowTopologyView } from "./WorkflowTopologyView";
import type { WorkflowTopology } from "@/lib/types";

function mockApi(response: unknown, status = 200) {
  global.fetch = vi.fn(() =>
    Promise.resolve(
      new Response(typeof response === "string" ? response : JSON.stringify(response), {
        status,
        headers: { "content-type": "application/json" },
      })
    )
  ) as unknown as typeof fetch;
}

const chainTopology: WorkflowTopology = {
  name: "refund-flow",
  type: "chain",
  nodes: [
    { id: "research", type: "agent", label: "researcher", model: "gpt-4o", tool_count: 2 },
    { id: "approval", type: "hitl", label: "Manager approval" },
    { id: "process", type: "tool", label: "process_refund", tool_name: "process_refund" },
  ],
  edges: [
    { from: "research", to: "approval", type: "sequential" },
    { from: "approval", to: "process", type: "conditional", condition: "approved == True" },
  ],
  entrypoint: "research",
  tools: [
    { owner: "research", name: "search_docs", type: "function" },
    { owner: "research", name: "lookup_customer", type: "function" },
  ],
  knowledge_bases: [],
};

describe("WorkflowTopologyView", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders one DOM node per topology node, tagged by type", async () => {
    mockApi(chainTopology);
    const { container } = renderWithProviders(
      <div style={{ width: 800, height: 600 }}>
        <WorkflowTopologyView runnerType="chain" name="refund-flow" />
      </div>
    );
    await waitFor(() =>
      expect(container.querySelector('[data-testid="workflow-topology"]')).toBeInTheDocument()
    );
    // Every topology node renders with its data-node-type for visual regression.
    await waitFor(() => {
      expect(container.querySelector('[data-node-type="agent"]')).toBeInTheDocument();
      expect(container.querySelector('[data-node-type="hitl"]')).toBeInTheDocument();
    });
    expect(screen.getByText("researcher")).toBeInTheDocument();
    expect(screen.getByText("Manager approval")).toBeInTheDocument();
    // process_refund appears twice (label + tool_name subtitle); both are valid.
    expect(screen.getAllByText("process_refund").length).toBeGreaterThan(0);
  });

  it("shows the registration callout when the topology endpoint 404s", async () => {
    mockApi({ detail: "not registered" }, 404);
    renderWithProviders(
      <WorkflowTopologyView runnerType="chain" name="missing" />
    );
    await waitFor(() =>
      expect(screen.getByText(/No topology available/i)).toBeInTheDocument()
    );
    // build_app(runners=…) appears in both the prose and the code snippet.
    expect(screen.getAllByText(/build_app\(runners=/i).length).toBeGreaterThan(0);
  });

  it("renders the layout toggle in non-compact mode", async () => {
    mockApi(chainTopology);
    renderWithProviders(
      <div style={{ width: 800, height: 600 }}>
        <WorkflowTopologyView runnerType="chain" name="refund-flow" />
      </div>
    );
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /Horizontal/i })).toBeInTheDocument()
    );
    expect(screen.getByRole("button", { name: /Vertical/i })).toBeInTheDocument();
  });

  it("hides the layout toggle in compact preview mode", async () => {
    mockApi(chainTopology);
    renderWithProviders(
      <div style={{ width: 400, height: 200 }}>
        <WorkflowTopologyView runnerType="chain" name="refund-flow" compact height={180} />
      </div>
    );
    await waitFor(() =>
      expect(screen.getByText("researcher")).toBeInTheDocument()
    );
    expect(screen.queryByRole("button", { name: /Horizontal/i })).toBeNull();
  });
});
