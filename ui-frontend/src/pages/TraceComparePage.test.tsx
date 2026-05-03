import { describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "@/test/utils";
import { TraceComparePage } from "./TraceComparePage";
import type { CompareTracesResponse } from "@/lib/types";

/**
 * Smoke tests for the trace comparison page.
 *
 * Per the Sprint 3 decisions: the only mock surface is the fetch boundary
 * (api.ts). The summary-card math, alignment-row classification, and
 * react-diff-viewer rendering are exercised against the real components.
 */

function fixture(overrides?: Partial<CompareTracesResponse>): CompareTracesResponse {
  const base: CompareTracesResponse = {
    trace_a: {
      trace_id: "trace-a",
      name: "demo agent",
      status: "OK",
      start_time: new Date(Date.now() - 60_000).toISOString(),
      end_time: new Date(Date.now() - 58_000).toISOString(),
      agent_name: "demo",
      thread_id: null,
      total_cost_usd: 0.04,
      total_tokens: 312,
      span_count: 3,
      duration_ms: 2460,
      runner_type: "agent",
      runner_name: "demo",
      spans: [
        {
          span_id: "a-0",
          trace_id: "trace-a",
          parent_span_id: null,
          name: "agent.demo",
          start_time: new Date().toISOString(),
          end_time: new Date().toISOString(),
          status: "OK",
          attributes: {},
          events: [],
        },
      ],
    },
    trace_b: {
      trace_id: "trace-b",
      name: "demo agent",
      status: "OK",
      start_time: new Date().toISOString(),
      end_time: new Date().toISOString(),
      agent_name: "demo",
      thread_id: null,
      total_cost_usd: 0.07,
      total_tokens: 487,
      span_count: 4,
      duration_ms: 3120,
      runner_type: "agent",
      runner_name: "demo",
      spans: [
        {
          span_id: "b-0",
          trace_id: "trace-b",
          parent_span_id: null,
          name: "agent.demo",
          start_time: new Date().toISOString(),
          end_time: new Date().toISOString(),
          status: "OK",
          attributes: {},
          events: [],
        },
      ],
    },
    alignment: [
      {
        index: 0,
        span_a: {
          span_id: "a-0",
          name: "agent.demo",
          status: "OK",
          start_time: "",
          end_time: "",
          duration_ms: 100,
        },
        span_b: {
          span_id: "b-0",
          name: "agent.demo",
          status: "OK",
          start_time: "",
          end_time: "",
          duration_ms: 700,
        },
        match: "slower",
        delta_ms: 600,
      },
      {
        index: 1,
        span_a: null,
        span_b: {
          span_id: "b-1",
          name: "tool.validate_input",
          status: "OK",
          start_time: "",
          end_time: "",
          duration_ms: 50,
        },
        match: "new_in_b",
        delta_ms: null,
      },
    ],
    summary: {
      duration_delta_ms: 660,
      tokens_delta: 175,
      cost_delta_usd: 0.03,
      spans_delta: 1,
      time_apart_seconds: 259_200,
    },
  };
  return { ...base, ...overrides };
}

function mockFetchOnce(payload: unknown) {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(
    new Response(JSON.stringify(payload), {
      status: 200,
      headers: { "content-type": "application/json" },
    })
  );
}

describe("TraceComparePage", () => {
  it("renders the empty state when no ?a=&b= is in the URL", () => {
    renderWithProviders(<TraceComparePage />, { route: "/traces/compare" });
    expect(screen.getByText("Two trace IDs required")).toBeInTheDocument();
  });

  it("renders summary deltas, alignment table, and time-apart label from the API payload", async () => {
    mockFetchOnce(fixture());
    renderWithProviders(<TraceComparePage />, {
      route: "/traces/compare?a=trace-a&b=trace-b",
    });

    // Time apart: 259200s == 3 days
    await screen.findByText(/3d apart/);

    // Summary card row is rendered.
    expect(screen.getByTestId("trace-compare-summary")).toBeInTheDocument();

    // Alignment table rendered with both rows.
    expect(screen.getByTestId("span-alignment-table")).toBeInTheDocument();
    // "agent.demo" shows once per side of the row, plus once in the
    // header label — getAllByText catches all and asserts at least one.
    expect(screen.getAllByText("agent.demo").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("tool.validate_input")).toBeInTheDocument();
    // Match badge for the slower row.
    expect(screen.getByText("slower")).toBeInTheDocument();
    // The "new in B" row also has its badge.
    expect(screen.getByText("new in B")).toBeInTheDocument();
  });

  it("expands a row to show the input/output/attributes diff blocks", async () => {
    mockFetchOnce(fixture());
    renderWithProviders(<TraceComparePage />, {
      route: "/traces/compare?a=trace-a&b=trace-b",
    });

    const expandBtn = await screen.findAllByRole("button", {
      name: /expand span diff/i,
    });
    const user = userEvent.setup();
    await user.click(expandBtn[0]);

    // The three DiffBlock headers should now be visible.
    await waitFor(() => {
      expect(screen.getByText("Input")).toBeInTheDocument();
      expect(screen.getByText("Output")).toBeInTheDocument();
      expect(screen.getByText("Attributes")).toBeInTheDocument();
    });
  });
});
