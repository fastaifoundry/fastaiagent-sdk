import { beforeEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { WorkflowsPage } from "./WorkflowsPage";

function mockApi(response: unknown) {
  global.fetch = vi.fn(() =>
    Promise.resolve(
      new Response(JSON.stringify(response), {
        status: 200,
        headers: { "content-type": "application/json" },
      })
    )
  ) as unknown as typeof fetch;
}

describe("WorkflowsPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("shows an empty state when no workflows have run", async () => {
    mockApi({ workflows: [] });
    renderWithProviders(<WorkflowsPage />);
    await waitFor(() =>
      expect(screen.getByText(/No workflows yet/i)).toBeInTheDocument()
    );
  });

  it("renders a card per workflow with the runner-type chip", async () => {
    mockApi({
      workflows: [
        {
          runner_type: "chain",
          workflow_name: "support-flow",
          run_count: 5,
          success_rate: 0.8,
          error_count: 1,
          avg_latency_ms: 1200,
          avg_cost_usd: 0.002,
          last_run: new Date().toISOString(),
          node_count: 3,
        },
        {
          runner_type: "swarm",
          workflow_name: "research-team",
          run_count: 2,
          success_rate: 1,
          error_count: 0,
          avg_latency_ms: 4500,
          avg_cost_usd: 0.01,
          last_run: new Date().toISOString(),
          node_count: null,
        },
      ],
    });
    renderWithProviders(<WorkflowsPage />);
    await waitFor(() =>
      expect(screen.getByText("support-flow")).toBeInTheDocument()
    );
    expect(screen.getByText("research-team")).toBeInTheDocument();
    // Card chips show the runner type. Both tab labels and card chips
    // contain the runner-type word, so scope to exact-text chip lookup.
    expect(screen.getByText(/chain · 3 nodes/i)).toBeInTheDocument();
    expect(screen.getByText("swarm", { exact: true })).toBeInTheDocument();
  });
});
