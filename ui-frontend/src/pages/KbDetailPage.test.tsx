/**
 * The KB search tab picks a search type the KB can serve (1.89.0).
 *
 * A keyword-only KB was searched, and labelled, "hybrid". The detail now says
 * which types work; the tab offers those, defaults to the KB's own, shows the
 * type the server used, and shows the 400 for one it can't. fetch is the only
 * mock.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { Route, Routes } from "react-router-dom";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { KbDetailPage } from "./KbDetailPage";

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function stub(detail: Record<string, unknown>, searchStatus = 200) {
  const searches: Record<string, unknown>[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const path = String(input).split("?")[0];
    if (path === "/api/kb/docs/search") {
      const body = JSON.parse(String(init?.body));
      searches.push(body);
      if (searchStatus !== 200) {
        return json(
          { detail: "KB 'docs' has no embeddings, so it cannot serve a vector search. Use search_type='keyword'." },
          searchStatus
        );
      }
      return json({ query: body.query, top_k: body.top_k, search_type: body.search_type ?? "keyword", results: [] });
    }
    if (path === "/api/kb/docs") {
      return json({
        name: "docs",
        path: "/tmp/docs",
        chunk_count: 2,
        doc_count: 1,
        size_bytes: 10,
        last_updated: "2026-10-14T00:00:00Z",
        metadata_keys: [],
        ...detail,
      });
    }
    if (path === "/api/kb/docs/documents") return json({ total: 0, page: 1, page_size: 100, documents: [] });
    return json({});
  });
  return searches;
}

async function openSearch() {
  renderWithProviders(
    <Routes>
      <Route path="/kb/:name" element={<KbDetailPage />} />
    </Routes>,
    { route: "/kb/docs" }
  );
  const tab = await screen.findByRole("tab", { name: /search playground/i });
  fireEvent.mouseDown(tab, { button: 0 });
  await screen.findByLabelText("Query");
}

afterEach(() => vi.restoreAllMocks());

describe("KbDetailPage search", () => {
  it("offers only keyword search on a KB without embeddings", async () => {
    const searches = stub({ has_embeddings: false, search_types: ["keyword"], search_type: "keyword" });
    await openSearch();
    expect(screen.getByTestId("kb-keyword-only")).toBeInTheDocument();
    const select = screen.getByLabelText("Search type") as HTMLSelectElement;
    expect(Array.from(select.options).map((o) => o.value)).toEqual(["keyword"]);

    fireEvent.change(screen.getByLabelText("Query"), { target: { value: "refund" } });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(searches).toHaveLength(1));
    expect(searches[0]).toMatchObject({ query: "refund", search_type: "keyword" });
    // The type the server actually used.
    expect(await screen.findByText(/0 results/)).toHaveTextContent("search_type = keyword");
  });

  it("defaults to hybrid with embeddings and sends the type picked", async () => {
    const searches = stub({
      has_embeddings: true,
      search_types: ["keyword", "vector", "hybrid"],
      search_type: "hybrid",
    });
    await openSearch();
    const select = screen.getByLabelText("Search type") as HTMLSelectElement;
    expect(select.value).toBe("hybrid");
    fireEvent.change(select, { target: { value: "vector" } });
    fireEvent.change(screen.getByLabelText("Query"), { target: { value: "refund" } });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(searches[0]).toMatchObject({ search_type: "vector" }));
  });

  it("shows the server's 400 for a type the KB cannot serve", async () => {
    stub({ has_embeddings: true, search_types: ["keyword", "vector", "hybrid"], search_type: "hybrid" }, 400);
    await openSearch();
    fireEvent.change(screen.getByLabelText("Query"), { target: { value: "refund" } });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("has no embeddings");
  });
});
