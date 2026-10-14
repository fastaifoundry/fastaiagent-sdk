/**
 * The Traces list's URL filters reach the API (1.89.0): `source`, `noted`,
 * and `production_only` (the server's other name for `exclude_experiments`).
 * fetch is the only mock.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { TracesPage } from "./TracesPage";

function stub() {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    if (String(input).startsWith("/api/filter-presets")) {
      return new Response("[]", { status: 200, headers: { "content-type": "application/json" } });
    }
    const body = {
      rows: [
        {
          trace_id: "b".repeat(32),
          name: "playground.run",
          start_time: "2026-10-14T00:00:00Z",
          end_time: "2026-10-14T00:00:01Z",
          status: "OK",
          span_count: 2,
          duration_ms: 1000,
          agent_name: null,
          thread_id: null,
          total_cost_usd: null,
          total_tokens: null,
          runner_type: "playground",
          runner_name: null,
          source: "playground",
          noted: true,
        },
      ],
      total: 1,
      page: 1,
      page_size: 100,
    };
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  });
}

afterEach(() => vi.restoreAllMocks());

describe("TracesPage", () => {
  it("sends source, noted and production-only from the URL", async () => {
    const spy = stub();
    renderWithProviders(<TracesPage />, {
      route: "/traces?source=playground&noted=true&production_only=true",
    });
    await waitFor(() => {
      const url = spy.mock.calls
        .map(([u]) => new URL(String(u), "http://x"))
        .find((u) => u.pathname === "/api/traces" && u.searchParams.get("page_size") === "100");
      expect(url?.searchParams.get("source")).toBe("playground");
      expect(url?.searchParams.get("noted")).toBe("true");
      expect(url?.searchParams.get("exclude_experiments")).toBe("true");
    });
  });

  it("drops a source the server would reject", async () => {
    const spy = stub();
    renderWithProviders(<TracesPage />, { route: "/traces?source=%25drop" });
    await screen.findByText("playground.run");
    const urls = spy.mock.calls.map(([u]) => new URL(String(u), "http://x"));
    expect(urls.every((u) => !u.searchParams.has("source"))).toBe(true);
  });

  it("marks a noted trace and badges a Playground root", async () => {
    stub();
    renderWithProviders(<TracesPage />, { route: "/traces" });
    expect(await screen.findByTestId("trace-noted")).toBeInTheDocument();
    expect(screen.getAllByText("playground").length).toBeGreaterThan(0);
  });
});
