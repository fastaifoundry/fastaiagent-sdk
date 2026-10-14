/**
 * The execution and approval pages say what happened (1.89.0, A1).
 *
 * Every interrupted checkpoint used to read "waiting", forever; the page had
 * no outcome; a resume left the approvals list stale; the reason only went in
 * metadata. fetch is the only mock.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { Route, Routes } from "react-router-dom";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { ExecutionPage } from "./ExecutionPage";
import { ApprovalDetailPage } from "./ApprovalDetailPage";

const EID = "exec-1";
const RUN_TRACE = "1".repeat(32);
const RESUME_TRACE = "2".repeat(32);

function checkpoint(i: number, status: string, extra: Record<string, unknown> = {}) {
  return {
    checkpoint_id: `cp${i}`,
    parent_checkpoint_id: i ? `cp${i - 1}` : null,
    chain_name: "refund-agent",
    execution_id: EID,
    node_id: i === 1 ? "approve_refund" : `step${i}`,
    node_index: i,
    status,
    state_snapshot: {},
    node_input: {},
    node_output: {},
    iteration: 0,
    iteration_counters: {},
    interrupt_reason: status === "interrupted" ? "policy_approval_required" : null,
    interrupt_context: {},
    agent_path: "refund-agent",
    created_at: "2026-10-14T10:00:00Z",
    synced: null,
    sync_error: null,
    ...extra,
  };
}

const PENDING = {
  execution_id: EID,
  chain_name: "refund-agent",
  node_id: "approve_refund",
  reason: "policy_approval_required",
  context: { amount: 50 },
  agent_path: "refund-agent",
  created_at: "2026-10-14T10:00:00Z",
};

const RESOLVED = {
  execution_id: EID,
  chain_name: "refund-agent",
  status: "completed",
  agent_path: "refund-agent",
  checkpoint_count: 3,
  latest_checkpoint_id: "cp2",
  latest_state_snapshot: {},
  checkpoints: [
    checkpoint(0, "completed"),
    checkpoint(1, "interrupted", { interrupt_state: "resolved" }),
    checkpoint(2, "completed"),
  ],
  pending_interrupt: null,
  runs: [
    { trace_id: RUN_TRACE, role: "run", name: "agent.refund-agent", started_at: "2026-10-14T10:00:00Z", ended_at: null, status: "paused", output: null, resume: null },
    {
      trace_id: RESUME_TRACE,
      role: "resume",
      name: "agent.refund-agent",
      started_at: "2026-10-14T10:05:00Z",
      ended_at: null,
      status: "completed",
      output: "Refund issued.",
      resume: { node_id: "approve_refund", kind: "approval", approved: true, resolver: "alice", reason: "within policy" },
    },
  ],
  resolution: {
    approved: true,
    resolver: "alice",
    reason: "within policy",
    at: "2026-10-14T10:05:00Z",
    trace_id: RESUME_TRACE,
    source: "trace",
  },
  final_output: "Refund issued.",
};

const WAITING = {
  ...RESOLVED,
  status: "interrupted",
  checkpoint_count: 2,
  latest_checkpoint_id: "cp1",
  checkpoints: [checkpoint(0, "completed"), checkpoint(1, "interrupted", { interrupt_state: "pending" })],
  pending_interrupt: PENDING,
  runs: [RESOLVED.runs[0]],
  resolution: null,
  final_output: null,
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function stub(execution: unknown, pendingItems: unknown[] = []) {
  const posts: Record<string, unknown>[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const path = String(input).split("?")[0];
    if ((init?.method ?? "GET").toUpperCase() === "POST" && path === `/api/executions/${EID}/resume`) {
      posts.push(JSON.parse(String(init?.body)));
      return json({
        execution_id: EID,
        chain_name: "refund-agent",
        reason: "within policy",
        resolver: "alice",
        trace_id: RESUME_TRACE,
        result: { status: "completed", output: "Refund issued." },
      });
    }
    if (path === `/api/executions/${EID}`) return json(execution);
    if (path === `/api/executions/${EID}/idempotency-cache`) return json({ count: 0, items: [] });
    if (path === "/api/pending-interrupts") return json({ count: pendingItems.length, items: pendingItems });
    return json({});
  });
  return posts;
}

function renderAt(route: string) {
  return renderWithProviders(
    <Routes>
      <Route path="/executions/:execution_id" element={<ExecutionPage />} />
      <Route path="/approvals/:execution_id" element={<ApprovalDetailPage />} />
    </Routes>,
    { route }
  );
}

afterEach(() => vi.restoreAllMocks());

describe("ExecutionPage", () => {
  it("shows a resolved pause as resolved, with the decision and every run", async () => {
    stub(RESOLVED);
    renderAt(`/executions/${EID}`);
    const outcome = await screen.findByTestId("execution-outcome");
    expect(within(outcome).getByTestId("execution-resolution")).toHaveTextContent("Approved");
    expect(within(outcome).getByTestId("execution-resolution")).toHaveTextContent("alice");
    expect(within(outcome).getByText("within policy")).toBeInTheDocument();
    expect(within(outcome).getByTestId("execution-final-output")).toHaveTextContent("Refund issued.");

    const runs = within(outcome).getByTestId("execution-runs");
    const links = within(runs).getAllByRole("link");
    expect(links.map((a) => a.getAttribute("href"))).toEqual([
      `/traces/${RUN_TRACE}`,
      `/traces/${RESUME_TRACE}`,
    ]);
    expect(runs).toHaveTextContent("resume");
    expect(runs).toHaveTextContent("paused");

    expect(screen.getByTestId("interrupt-resolved")).toHaveTextContent("resolved · approved by alice");
    expect(screen.queryByTestId("interrupt-waiting")).toBeNull();
    expect(screen.queryByRole("link", { name: "Open approval" })).toBeNull();
  });

  it("still shows a pending pause as waiting, with the approval link", async () => {
    stub(WAITING, [PENDING]);
    renderAt(`/executions/${EID}`);
    expect(await screen.findByTestId("interrupt-waiting")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open approval" })).toBeInTheDocument();
    expect(screen.getByTestId("execution-outcome")).toHaveTextContent("Waiting for a decision");
  });
});

describe("ApprovalDetailPage", () => {
  it("sends the reason top-level, then links the resumed run's trace", async () => {
    const posts = stub(RESOLVED, [PENDING]);
    const { queryClient } = renderAt(`/approvals/${EID}`);
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");
    const box = await screen.findByPlaceholderText(/Why are you approving/);
    fireEvent.change(box, { target: { value: "within policy" } });
    fireEvent.click(screen.getByRole("button", { name: /approve/i }));

    await waitFor(() => expect(posts).toHaveLength(1));
    expect(posts[0]).toEqual({
      approved: true,
      reason: "within policy",
      metadata: { reason: "within policy" },
    });
    const banner = await screen.findByTestId("resumed-banner");
    expect(within(banner).getByRole("link", { name: /view trace/i }).getAttribute("href")).toBe(
      `/traces/${RESUME_TRACE}`
    );
    // Every view of the pause is stale once it is resolved.
    const keys = invalidate.mock.calls.map(([f]) => JSON.stringify(f?.queryKey));
    expect(keys).toEqual(
      expect.arrayContaining([
        JSON.stringify(["execution", EID]),
        JSON.stringify(["pending-interrupts"]),
        JSON.stringify(["pending-for-execution", EID]),
        JSON.stringify(["traces"]),
      ])
    );
  });

  it("says who resolved a pause that no longer waits", async () => {
    stub(RESOLVED, []);
    renderAt(`/approvals/${EID}`);
    const card = await screen.findByTestId("approval-resolved");
    expect(card).toHaveTextContent("Approved by alice");
    expect(card).toHaveTextContent("within policy");
    expect(screen.queryByText(/may have already been resumed/)).toBeNull();
  });
});
