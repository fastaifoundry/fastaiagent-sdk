import { describe, expect, it, vi } from "vitest";
import { Routes, Route } from "react-router-dom";
import { screen } from "@testing-library/react";
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
});
