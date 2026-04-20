import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { TracesTable } from "./TracesTable";
import type { TraceRow } from "@/lib/types";

const SAMPLE: TraceRow[] = [
  {
    trace_id: "abcdef0123456789abcdef0123456789",
    name: "agent.support-bot",
    start_time: new Date(Date.now() - 5 * 60_000).toISOString(),
    end_time: new Date(Date.now() - 4 * 60_000).toISOString(),
    status: "OK",
    span_count: 4,
    duration_ms: 1234,
    agent_name: "support-bot",
    thread_id: null,
    total_cost_usd: 0.0042,
    total_tokens: 512,
    runner_type: "agent",
    runner_name: "support-bot",
  },
  {
    trace_id: "failing01234567890abc",
    name: "agent.flaky",
    start_time: new Date(Date.now() - 10 * 60_000).toISOString(),
    end_time: null,
    status: "ERROR",
    span_count: 2,
    duration_ms: null,
    agent_name: "flaky",
    thread_id: null,
    total_cost_usd: null,
    total_tokens: null,
    runner_type: "agent",
    runner_name: "flaky",
  },
];

describe("TracesTable", () => {
  it("renders every row with formatted numbers and truncated id", () => {
    renderWithProviders(<TracesTable rows={SAMPLE} />);
    expect(screen.getByText("agent.support-bot")).toBeInTheDocument();
    expect(screen.getByText("agent.flaky")).toBeInTheDocument();

    // Status badges picked up the right labels.
    expect(screen.getAllByText("OK")).toHaveLength(1);
    expect(screen.getAllByText("ERROR")).toHaveLength(1);

    // Short id appears for each trace.
    expect(screen.getByText(/abcdef01…6789/)).toBeInTheDocument();
    expect(screen.getByText(/failing0…0abc/)).toBeInTheDocument();

    // Numeric cells formatted.
    expect(screen.getByText("1.23s")).toBeInTheDocument();
    expect(screen.getByText("512")).toBeInTheDocument();
    expect(screen.getByText("$0.0042")).toBeInTheDocument();

    // Null-safe cells.
    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThanOrEqual(3);
  });

  it("renders column headers", () => {
    renderWithProviders(<TracesTable rows={SAMPLE} />);
    expect(screen.getByRole("columnheader", { name: "Trace" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Name" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Workflow" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Runner" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Thread" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Status" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Cost" })).toBeInTheDocument();
  });

  it("renders a clickable thread chip when thread_id is present", () => {
    renderWithProviders(
      <TracesTable
        rows={[
          { ...SAMPLE[0], thread_id: "session-demo" },
          SAMPLE[1],
        ]}
      />
    );
    const threadLink = screen.getByRole("link", { name: /session-demo/i });
    expect(threadLink).toHaveAttribute("href", "/threads/session-demo");
  });
});
