import { describe, expect, it, vi } from "vitest";
import { Routes, Route } from "react-router-dom";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { DatasetDetailPage } from "./DatasetDetailPage";
import type { DatasetDetail } from "@/lib/types";

/**
 * Smoke tests for the dataset detail page. The api.ts boundary is the
 * only mock surface, per the Sprint 3 testing decision.
 */

function fixture(overrides?: Partial<DatasetDetail>): DatasetDetail {
  const base: DatasetDetail = {
    name: "echo",
    cases: [
      {
        index: 0,
        input: "Reply with the word 'hello'.",
        expected_output: "hello",
        tags: ["smoke"],
        metadata: {},
        extra: {},
      },
      {
        index: 1,
        input: [
          { type: "text", text: "What animal is this?" },
          { type: "image", path: "images/echo/cat.png" },
        ],
        expected_output: "cat",
        tags: ["vision"],
        metadata: {},
        extra: { source_trace_id: "t-1" },
      },
    ],
  };
  return { ...base, ...overrides };
}

function mockFetch(payload: unknown) {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(
    new Response(JSON.stringify(payload), {
      status: 200,
      headers: { "content-type": "application/json" },
    })
  );
}

describe("DatasetDetailPage", () => {
  it("renders the cases table with text + multimodal previews", async () => {
    mockFetch(fixture());
    renderWithProviders(
      <Routes>
        <Route path="/datasets/:name" element={<DatasetDetailPage />} />
      </Routes>,
      { route: "/datasets/echo" }
    );

    expect(await screen.findByText(/Reply with the word 'hello'/)).toBeInTheDocument();
    expect(screen.getByText(/What animal is this/)).toBeInTheDocument();
    // Multimodal badge shows on the second row
    expect(screen.getByText("image")).toBeInTheDocument();
    // Tags column rendered for both cases
    expect(screen.getByText("smoke")).toBeInTheDocument();
    expect(screen.getByText("vision")).toBeInTheDocument();
  });

  it("renders an empty state when the dataset has no cases", async () => {
    mockFetch(fixture({ cases: [] }));
    renderWithProviders(
      <Routes>
        <Route path="/datasets/:name" element={<DatasetDetailPage />} />
      </Routes>,
      { route: "/datasets/echo" }
    );

    expect(await screen.findByText("No cases yet")).toBeInTheDocument();
  });

  it("'Run eval' opens the dialog and runs nothing until asked (1.89.0)", async () => {
    const calls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      calls.push(`${(init?.method ?? "GET").toUpperCase()} ${url}`);
      const body = url.endsWith("/run-eval/options")
        ? {
            agents: [],
            scorers: [],
            judge_models: [],
            dataset_features: {
              rows: 2,
              with_expected_output: 2,
              with_expected_trajectory: 0,
              with_expected_tool_calls: 0,
              with_expected_steps: 0,
              with_context: 0,
              multimodal: 1,
            },
            echo_available: true,
          }
        : fixture();
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    });
    renderWithProviders(
      <Routes>
        <Route path="/datasets/:name" element={<DatasetDetailPage />} />
      </Routes>,
      { route: "/datasets/echo" }
    );

    fireEvent.click(await screen.findByRole("button", { name: /run eval/i }));
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByRole("heading", { name: "Run eval" })).toBeInTheDocument();
    expect(
      await within(dialog).findByRole("radio", { name: /dataset check/i })
    ).toBeChecked();
    expect(calls).toContain("GET /api/datasets/echo/run-eval/options");
    expect(calls.some((c) => c.startsWith("POST"))).toBe(false);
  });

  it("Duplicate keeps the row's other fields", async () => {
    const posts: Record<string, unknown>[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      if ((init?.method ?? "GET").toUpperCase() === "POST" && url.endsWith("/cases")) {
        posts.push(JSON.parse(String(init?.body)));
      }
      return new Response(JSON.stringify(fixture()), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    });
    renderWithProviders(
      <Routes>
        <Route path="/datasets/:name" element={<DatasetDetailPage />} />
      </Routes>,
      { route: "/datasets/echo" }
    );

    await screen.findByText(/What animal is this/);
    fireEvent.click(screen.getAllByTitle("Duplicate")[1]!);
    await waitFor(() => expect(posts).toHaveLength(1));
    expect(posts[0]!.extra).toEqual({ source_trace_id: "t-1" });
    // Let the mutation's invalidation refetch settle inside act().
    await waitFor(() => expect(screen.getAllByTitle("Duplicate")[1]).toBeEnabled());
  });

  it("disables 'Run eval' on an empty dataset", async () => {
    mockFetch(fixture({ cases: [] }));
    renderWithProviders(
      <Routes>
        <Route path="/datasets/:name" element={<DatasetDetailPage />} />
      </Routes>,
      { route: "/datasets/echo" }
    );
    expect(await screen.findByRole("button", { name: /run eval/i })).toBeDisabled();
  });
});
