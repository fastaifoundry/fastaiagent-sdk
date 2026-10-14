/**
 * Component tests for AgentDependencyGraph.
 *
 * Same setup as WorkflowTopologyView.test.tsx: jsdom doesn't paint, so the
 * parent gets an inline size and assertions read data-attrs.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { waitFor } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { AgentDependencyGraph } from "./AgentDependencyGraph";
import type { AgentDependencies } from "@/lib/types";

function mockApi(response: unknown, status = 200) {
  global.fetch = vi.fn(() =>
    Promise.resolve(
      new Response(JSON.stringify(response), {
        status,
        headers: { "content-type": "application/json" },
      })
    )
  ) as unknown as typeof fetch;
}

const deps: AgentDependencies = {
  agent: { name: "support", type: "agent", model: "gpt-4o", provider: "openai" },
  tools: [],
  knowledge_bases: [],
  prompts: [],
  guardrails: [
    { name: "no_pii", guardrail_type: "code", position: "input", action: "block", blocking: true },
    { name: "no_pii", guardrail_type: "code", position: "output", action: "mask", blocking: true },
  ],
  observed_guardrails: [
    {
      name: "org_toxicity",
      position: "output",
      guardrail_type: "content_safety",
      last_seen: new Date().toISOString(),
      checks: 4,
      origin: "plane",
    },
  ],
  model: { provider: "openai", model: "gpt-4o" },
};

describe("AgentDependencyGraph", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("draws the same rule at two positions as two nodes", async () => {
    mockApi(deps);
    const { container } = renderWithProviders(
      <div style={{ width: 800, height: 600 }}>
        <AgentDependencyGraph agentName="support" />
      </div>
    );
    await waitFor(() =>
      expect(container.querySelector('[data-testid="agent-dependency-graph"]')).toBeInTheDocument()
    );
    await waitFor(() => {
      const declared = container.querySelectorAll(
        '[data-node-kind="guardrail"]:not([data-observed])'
      );
      expect(declared).toHaveLength(2);
    });
    const ids = Array.from(container.querySelectorAll(".react-flow__node-guardrail")).map((n) =>
      n.getAttribute("data-id")
    );
    expect(ids).toEqual(
      expect.arrayContaining([
        "guardrail:support:input:no_pii:0",
        "guardrail:support:output:no_pii:1",
      ])
    );
  });

  it("draws an observed rule dashed, with an observed badge", async () => {
    mockApi(deps);
    const { container } = renderWithProviders(
      <div style={{ width: 800, height: 600 }}>
        <AgentDependencyGraph agentName="support" />
      </div>
    );
    await waitFor(() =>
      expect(container.querySelector('[data-observed="true"]')).toBeInTheDocument()
    );
    const observed = container.querySelector('[data-observed="true"]')!;
    expect(observed.className).toMatch(/border-dashed/);
    expect(observed).toHaveTextContent("observed");
    expect(observed).toHaveTextContent("org_toxicity");
  });
});
