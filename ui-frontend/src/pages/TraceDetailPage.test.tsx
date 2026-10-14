/**
 * The trace page for a Playground run (1.89.0, P10g + notes).
 *
 * A `playground.run` root has no agent to replay, so "Open in Replay" is off;
 * its input and output show in the inspector like any other root's; the
 * trace's source and its note are on the page. fetch is the only mock.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { Route, Routes } from "react-router-dom";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { TraceDetailPage } from "./TraceDetailPage";

const TID = "a".repeat(32);

function root(attrs: Record<string, unknown>) {
  return {
    span_id: "s1",
    trace_id: TID,
    parent_span_id: null,
    name: "playground.run",
    start_time: "2026-10-14T00:00:00Z",
    end_time: "2026-10-14T00:00:01Z",
    status: "OK",
    attributes: attrs,
    events: [],
  };
}

function stub(detail: Record<string, unknown>) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = typeof input === "string" ? input : (input as Request).url;
    const path = url.split("?")[0];
    let body: unknown = {};
    if (path === `/api/traces/${TID}`) body = detail;
    else if (path === `/api/traces/${TID}/spans`) body = { tree: { span: (detail.spans as unknown[])[0], children: [] } };
    else if (path === "/api/traces") body = { rows: [], total: 0, page: 1, page_size: 50 };
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  });
}

function detail(extra: Record<string, unknown>) {
  const span = root({
    "fastaiagent.runner.type": "playground",
    "fastaiagent.playground.input": "What is 2+2?",
    "fastaiagent.playground.output": "4",
  });
  return {
    trace_id: TID,
    name: "playground.run",
    status: "OK",
    start_time: span.start_time,
    end_time: span.end_time,
    agent_name: null,
    thread_id: null,
    total_cost_usd: null,
    total_tokens: null,
    span_count: 1,
    runner_type: "playground",
    runner_name: null,
    spans: [span],
    source: "playground",
    note: null,
    ...extra,
  };
}

function renderPage() {
  renderWithProviders(
    <Routes>
      <Route path="/traces/:traceId" element={<TraceDetailPage />} />
    </Routes>,
    { route: `/traces/${TID}` }
  );
}

afterEach(() => vi.restoreAllMocks());

describe("TraceDetailPage", () => {
  it("disables Replay for a Playground trace and shows its source", async () => {
    stub(detail({}));
    renderPage();
    const button = await screen.findByRole("button", { name: /open in replay/i });
    expect(button).toBeDisabled();
    expect(screen.queryByRole("link", { name: /open in replay/i })).toBeNull();
    expect(screen.getByText(/source: playground/)).toBeInTheDocument();
  });

  it("shows the Playground root's input in the Input tab", async () => {
    stub(detail({}));
    renderPage();
    expect(await screen.findByText(/What is 2\+2\?/)).toBeInTheDocument();
  });

  it("keeps Replay for an agent trace", async () => {
    stub(detail({ runner_type: "agent", source: null, name: "agent.support" }));
    renderPage();
    expect(await screen.findByRole("link", { name: /open in replay/i })).toBeInTheDocument();
  });

  it("disables Replay for a KB search, whatever its root says", async () => {
    stub(detail({ runner_type: "agent", source: "kb_search" }));
    renderPage();
    expect(await screen.findByRole("button", { name: /open in replay/i })).toBeDisabled();
  });

  it("shows the trace's note", async () => {
    stub(detail({ note: { text: "bad refund", updated_at: "2026-10-14T00:00:00Z" } }));
    renderPage();
    expect(await screen.findByLabelText("Trace note")).toHaveValue("bad refund");
  });
});
