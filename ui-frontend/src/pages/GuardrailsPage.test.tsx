import { describe, expect, it, vi } from "vitest";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { useLocation } from "react-router-dom";
import { renderWithProviders } from "@/test/utils";
import { GuardrailsPage } from "./GuardrailsPage";
import type { GuardrailEvent } from "@/lib/types";

function row(i: number, overrides: Partial<GuardrailEvent> = {}): GuardrailEvent {
  return {
    event_id: `e${i}`,
    trace_id: `t${i}`,
    span_id: `s${i}`,
    guardrail_name: "no_pii",
    guardrail_type: "pii",
    position: "output",
    outcome: "blocked",
    score: null,
    message: null,
    agent_name: "support",
    timestamp: new Date().toISOString(),
    metadata: {},
    false_positive: false,
    false_positive_at: null,
    action: "block",
    action_taken: "blocked",
    severity: null,
    floor: false,
    effect: "blocked",
    final_reason: null,
    latency_ms: 1,
    attempt: null,
    batch_id: `b${i}`,
    guardrail_span_id: null,
    judged_ref: null,
    rule: null,
    blocking: true,
    on_error: "block",
    origin: "local",
    ...overrides,
  };
}

/** api.ts boundary (fetch) is the only mock surface; returns the page asked for. */
function mockEvents(total: number, extra: Record<string, unknown> = {}, rowsOverride?: GuardrailEvent[]) {
  return vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
    const url = new URL(String(input), "http://localhost");
    const page = Number(url.searchParams.get("page") ?? 1);
    const size = Number(url.searchParams.get("page_size") ?? 50);
    const start = (page - 1) * size;
    const rows =
      rowsOverride ??
      Array.from({ length: Math.max(0, Math.min(size, total - start)) }, (_, i) =>
        row(start + i)
      );
    return Promise.resolve(
      new Response(JSON.stringify({ rows, total, page, page_size: size, ...extra }), {
        status: 200,
        headers: { "content-type": "application/json" },
      })
    );
  });
}

function urls(spy: ReturnType<typeof mockEvents>): URL[] {
  return spy.mock.calls.map(([input]) => new URL(String(input), "http://localhost"));
}

function LocationProbe() {
  const loc = useLocation();
  return <div data-testid="location">{loc.search}</div>;
}

describe("GuardrailsPage", () => {
  it("hides passes by default", async () => {
    const spy = mockEvents(3);
    renderWithProviders(<GuardrailsPage />);
    await screen.findByTestId("guardrail-row-e0");
    expect(urls(spy)[0].searchParams.get("hide_passes")).toBe("true");

    // Turning the toggle off asks for passes too.
    fireEvent.click(screen.getByRole("switch", { name: "Hide passes" }));
    await waitFor(() => expect(urls(spy).at(-1)!.searchParams.has("hide_passes")).toBe(false));
  });

  it("pages with Next and Prev, and shows the range", async () => {
    const spy = mockEvents(120);
    renderWithProviders(<GuardrailsPage />);
    await screen.findByTestId("guardrail-row-e0");
    expect(screen.getByTestId("guardrail-pager-range")).toHaveTextContent("1–50 of 120");

    fireEvent.click(screen.getByRole("button", { name: /Next/ }));
    await screen.findByTestId("guardrail-row-e50");
    expect(urls(spy).at(-1)!.searchParams.get("page")).toBe("2");
    expect(screen.getByTestId("guardrail-pager-range")).toHaveTextContent("51–100 of 120");

    fireEvent.click(screen.getByRole("button", { name: /Prev/ }));
    await screen.findByTestId("guardrail-row-e0");
    expect(urls(spy).at(-1)!.searchParams.get("page")).toBe("1");
  });

  it("goes back to page 1 when a filter changes", async () => {
    const spy = mockEvents(120);
    renderWithProviders(<GuardrailsPage />);
    await screen.findByTestId("guardrail-row-e0");
    fireEvent.click(screen.getByRole("button", { name: /Next/ }));
    await screen.findByTestId("guardrail-row-e50");

    fireEvent.click(screen.getByRole("button", { name: "Hide experiments" }));
    await waitFor(() =>
      expect(urls(spy).at(-1)!.searchParams.get("exclude_experiments")).toBe("true")
    );
    expect(urls(spy).at(-1)!.searchParams.get("page")).toBe("1");
  });

  it("sends since/until from the date range", async () => {
    const spy = mockEvents(3);
    renderWithProviders(<GuardrailsPage />);
    await screen.findByTestId("guardrail-row-e0");

    // Quick range: since only.
    fireEvent.click(screen.getByRole("button", { name: "24h" }));
    await waitFor(() => expect(urls(spy).at(-1)!.searchParams.has("since")).toBe(true));
    expect(urls(spy).at(-1)!.searchParams.has("until")).toBe(false);

    // Custom range: both ends.
    fireEvent.click(screen.getByTitle("Custom date range"));
    const dialog = await screen.findByRole("dialog");
    const days = within(dialog)
      .getAllByRole("button")
      .filter((b) => /^\d+$/.test(b.textContent ?? ""));
    fireEvent.click(days[2]);
    fireEvent.click(days[5]);
    fireEvent.click(within(dialog).getByRole("button", { name: "Apply" }));
    await waitFor(() => {
      const last = urls(spy).at(-1)!;
      expect(last.searchParams.has("since")).toBe(true);
      expect(last.searchParams.has("until")).toBe(true);
    });
  });

  it("reads agent and rule from the URL", async () => {
    const spy = mockEvents(1);
    renderWithProviders(<GuardrailsPage />, {
      route: "/guardrails?agent=support&rule=no_pii",
    });
    await screen.findByTestId("guardrail-row-e0");
    expect(urls(spy)[0].searchParams.get("agent")).toBe("support");
    expect(urls(spy)[0].searchParams.get("rule")).toBe("no_pii");
  });

  it("writes filter changes and the page to the URL", async () => {
    mockEvents(120);
    renderWithProviders(
      <>
        <GuardrailsPage />
        <LocationProbe />
      </>,
      { route: "/guardrails?agent=support" }
    );
    await screen.findByTestId("guardrail-row-e0");
    const search = () => new URLSearchParams(screen.getByTestId("location").textContent ?? "");

    fireEvent.change(screen.getByDisplayValue("All outcomes"), { target: { value: "blocked" } });
    await waitFor(() => expect(search().get("outcome")).toBe("blocked"));
    expect(search().get("agent")).toBe("support");

    fireEvent.click(screen.getByRole("button", { name: "24h" }));
    await waitFor(() => expect(search().get("range")).toBe("24h"));

    fireEvent.click(screen.getByRole("switch", { name: "Hide passes" }));
    await waitFor(() => expect(search().get("hide_passes")).toBe("false"));

    fireEvent.click(screen.getByRole("button", { name: "Hide experiments" }));
    await waitFor(() => expect(search().get("exclude_experiments")).toBe("true"));

    await screen.findByTestId("guardrail-row-e0");
    fireEvent.click(screen.getByRole("button", { name: /Next/ }));
    await waitFor(() => expect(search().get("page")).toBe("2"));
  });

  it("restores filters and page from a shared URL", async () => {
    const spy = mockEvents(120);
    renderWithProviders(<GuardrailsPage />, {
      route: "/guardrails?outcome=errored&range=7d&hide_passes=false&exclude_experiments=true&fp=yes&page=2",
    });
    await waitFor(() => expect(spy).toHaveBeenCalled());
    const first = urls(spy)[0].searchParams;
    expect(first.get("outcome")).toBe("errored");
    expect(first.has("since")).toBe(true);
    expect(first.has("hide_passes")).toBe(false);
    expect(first.get("exclude_experiments")).toBe("true");
    expect(first.get("false_positive")).toBe("true");
    expect(first.get("page")).toBe("2");
    expect(screen.getByRole("button", { name: "7d" }).className).toMatch(/bg-primary/);
  });

  it("shows an observe-only mask and a refused rewrite in the Action cell", async () => {
    mockEvents(2, {}, [
      row(0, { outcome: "warned", effect: "observed", action: "mask", action_taken: "masked" }),
      row(1, {
        outcome: "blocked",
        effect: "blocked",
        action: "mask",
        action_taken: "masked",
        final_reason: "stream already yielded",
      }),
    ]);
    renderWithProviders(<GuardrailsPage />);
    await screen.findByTestId("guardrail-row-e0");
    expect(screen.getByText("masked · observe-only")).toBeInTheDocument();
    const refused = screen.getByText(/blocked \(was masked\)/);
    expect(refused).toHaveAttribute("title", "stream already yielded");
  });

  it("explains an empty list and points at legacy guardrail spans", async () => {
    mockEvents(0, { legacy_guardrail_spans: 3 });
    renderWithProviders(<GuardrailsPage />);

    expect(await screen.findByText("No guardrail checks recorded")).toBeInTheDocument();
    expect(screen.getByText(/FASTAIAGENT_TRACE_ENABLED/)).toBeInTheDocument();
    expect(screen.getByText(/FASTAIAGENT_LOCAL_DB/)).toBeInTheDocument();
    const legacy = screen.getByTestId("legacy-guardrail-spans");
    expect(legacy).toHaveTextContent(
      "3 guardrail spans exist in traces recorded before 1.89.0 — open Traces."
    );
    expect(within(legacy).getByRole("link", { name: "open Traces" })).toHaveAttribute(
      "href",
      "/traces"
    );
  });
});
