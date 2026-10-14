/**
 * An interrupted checkpoint is "waiting" only while the pause is pending
 * (1.89.0, A1). Real component, no mocks.
 */
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { CheckpointTimeline } from "./CheckpointTimeline";
import type { ExecutionCheckpoint } from "@/hooks/use-execution";

function interrupted(extra: Partial<ExecutionCheckpoint> = {}): ExecutionCheckpoint {
  return {
    checkpoint_id: "cp1",
    parent_checkpoint_id: null,
    chain_name: "refund-agent",
    execution_id: "e1",
    node_id: "approve_refund",
    node_index: 1,
    status: "interrupted",
    state_snapshot: {},
    node_input: {},
    node_output: {},
    iteration: 0,
    iteration_counters: {},
    interrupt_reason: "policy_approval_required",
    interrupt_context: {},
    agent_path: null,
    created_at: "2026-10-14T10:00:00Z",
    synced: null,
    sync_error: null,
    ...extra,
  };
}

describe("CheckpointTimeline interrupt rows", () => {
  it("says waiting for a pending pause", () => {
    render(<CheckpointTimeline checkpoints={[interrupted({ interrupt_state: "pending" })]} />);
    expect(screen.getByTestId("interrupt-waiting")).toHaveTextContent("waiting");
  });

  it("says resolved, and who rejected it", () => {
    render(
      <CheckpointTimeline
        checkpoints={[interrupted({ interrupt_state: "resolved" })]}
        resolution={{
          approved: false,
          resolver: "bob",
          reason: null,
          at: null,
          trace_id: null,
          source: "hitl_events",
        }}
      />
    );
    expect(screen.getByTestId("interrupt-resolved")).toHaveTextContent("resolved · rejected by bob");
    expect(screen.queryByTestId("interrupt-waiting")).toBeNull();
  });

  it("falls back to waiting when an older server doesn't say", () => {
    render(<CheckpointTimeline checkpoints={[interrupted()]} />);
    expect(screen.getByTestId("interrupt-waiting")).toBeInTheDocument();
  });
});
