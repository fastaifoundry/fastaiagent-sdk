import { beforeEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { KbListPage } from "./KbListPage";

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

describe("KbListPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("shows an empty state when no collections exist", async () => {
    mockApi({ root: "/tmp/kb", collections: [] });
    renderWithProviders(<KbListPage />);
    await waitFor(() =>
      expect(screen.getByText(/No LocalKB collections found/i)).toBeInTheDocument()
    );
  });

  it("renders a card per collection with stats", async () => {
    mockApi({
      root: "/tmp/kb",
      collections: [
        {
          name: "docs",
          path: "/tmp/kb/docs",
          chunk_count: 42,
          doc_count: 7,
          last_updated: new Date().toISOString(),
          size_bytes: 12345,
        },
      ],
    });
    renderWithProviders(<KbListPage />);
    await waitFor(() =>
      expect(screen.getByText("docs")).toBeInTheDocument()
    );
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("7")).toBeInTheDocument();
  });
});
