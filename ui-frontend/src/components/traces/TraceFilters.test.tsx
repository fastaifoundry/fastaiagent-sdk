import { describe, expect, it, vi } from "vitest";
import { useState } from "react";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { TraceFiltersBar } from "./TraceFilters";
import type { TraceFilters } from "@/lib/types";

/**
 * Smoke tests for the Sprint 3 enhancements to the trace filter bar.
 * Render + low-level fireEvent interactions only — heavier
 * userEvent / animation flows are covered by the Sprint 3 Playwright
 * spec, which runs against a real browser and a live server.
 *
 * Per the agreed Sprint 3 testing convention, the api.ts boundary is
 * the only mock surface; the components themselves render for real.
 */

function HostedFilterBar({ initial }: { initial?: TraceFilters }) {
  const [filters, setFilters] = useState<TraceFilters>(
    initial ?? { page: 1, page_size: 100 }
  );
  return (
    <div>
      <TraceFiltersBar filters={filters} onChange={setFilters} />
      <pre data-testid="filters-state">{JSON.stringify(filters)}</pre>
    </div>
  );
}

function mockEmptyFetch() {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(
    new Response("[]", {
      status: 200,
      headers: { "content-type": "application/json" },
    })
  );
}

describe("TraceFiltersBar (Sprint 3)", () => {
  it("renders the new 30d quick-range button", () => {
    mockEmptyFetch();
    renderWithProviders(<HostedFilterBar />);
    expect(screen.getByRole("button", { name: "30d" })).toBeInTheDocument();
  });

  it("renders the Custom date-range trigger", () => {
    mockEmptyFetch();
    renderWithProviders(<HostedFilterBar />);
    expect(screen.getByRole("button", { name: /Custom/i })).toBeInTheDocument();
  });

  it("renders the More filters disclosure (collapsed by default)", () => {
    mockEmptyFetch();
    renderWithProviders(<HostedFilterBar />);
    expect(
      screen.getByRole("button", { name: /More filters/i })
    ).toBeInTheDocument();
    // Collapsed: the labels inside the panel are not in the DOM.
    expect(screen.queryByText("Duration (ms)")).not.toBeInTheDocument();
  });

  it("opens the disclosure when the toggle is fireEvent-clicked", () => {
    mockEmptyFetch();
    renderWithProviders(<HostedFilterBar />);
    const toggle = screen.getByRole("button", { name: /More filters/i });
    fireEvent.click(toggle);
    expect(screen.getByText("Duration (ms)")).toBeInTheDocument();
    expect(screen.getByText("Cost (USD)")).toBeInTheDocument();
  });

  it("renders the Save preset button next to the presets dropdown", () => {
    mockEmptyFetch();
    renderWithProviders(<HostedFilterBar />);
    expect(
      screen.getByRole("button", { name: /Save preset/i })
    ).toBeInTheDocument();
  });

  it("debounces the search input — input updates immediately, filters.q after ~300ms", async () => {
    mockEmptyFetch();
    renderWithProviders(<HostedFilterBar />);
    const search = screen.getByPlaceholderText(
      /Search trace name/
    ) as HTMLInputElement;

    fireEvent.change(search, { target: { value: "refund" } });

    // Local input mirrors the keystroke immediately.
    expect(search.value).toBe("refund");
    // ...but filters.q hasn't propagated yet.
    expect(screen.getByTestId("filters-state").textContent).not.toContain(
      '"q":"refund"'
    );

    // After the 300ms debounce, parent state updates.
    await waitFor(
      () =>
        expect(screen.getByTestId("filters-state").textContent).toContain(
          '"q":"refund"'
        ),
      { timeout: 1000 }
    );
  });

  it("clears all active filters when the Clear button is pressed", () => {
    mockEmptyFetch();
    renderWithProviders(
      <HostedFilterBar
        initial={{
          status: "ERROR",
          agent: "demo",
          min_cost: 0.05,
          page: 1,
          page_size: 100,
        }}
      />
    );

    const clear = screen.getByRole("button", { name: /Clear/i });
    fireEvent.click(clear);

    const state = screen.getByTestId("filters-state").textContent ?? "";
    expect(state).not.toContain('"status":"ERROR"');
    expect(state).not.toContain('"agent":"demo"');
    expect(state).not.toContain('"min_cost"');
  });
});
