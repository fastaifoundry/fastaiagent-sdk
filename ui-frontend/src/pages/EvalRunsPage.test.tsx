import { describe, expect, it, vi } from "vitest";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { useLocation } from "react-router-dom";
import { renderWithProviders } from "@/test/utils";
import { EvalRunsPage } from "./EvalRunsPage";
import type { EvalRunRow } from "@/lib/types";

function row(i: number, over: Partial<EvalRunRow> = {}): EvalRunRow {
  return {
    run_id: `r${i}`,
    run_name: `run ${i}`,
    dataset_name: "qa",
    agent_name: "support",
    agent_version: null,
    scorers: ["exact_match"],
    started_at: "2026-10-14T10:00:00+00:00",
    finished_at: null,
    pass_count: 8,
    fail_count: 2,
    pass_rate: 0.8,
    metadata: { gate_outcome: "failed", git_branch: "main" },
    kind: "pytest",
    summary_version: 2,
    case_count: 6,
    case_pass_count: 3,
    case_fail_count: 1,
    case_error_count: 2,
    case_pass_rate: 0.75,
    agent_cost_usd: null,
    agent_cost_known: false,
    ...over,
  };
}

/** api.ts boundary (fetch) is the only mock surface. */
function mockRuns(total: number) {
  return vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
    const url = new URL(String(input), "http://localhost");
    if (url.pathname.endsWith("/evals/trend")) {
      return Promise.resolve(
        new Response(JSON.stringify({ points: [] }), {
          status: 200,
          headers: { "content-type": "application/json" },
        })
      );
    }
    const page = Number(url.searchParams.get("page") ?? 1);
    const size = Number(url.searchParams.get("page_size") ?? 50);
    const start = (page - 1) * size;
    const rows = Array.from({ length: Math.max(0, Math.min(size, total - start)) }, (_, i) =>
      row(start + i)
    );
    return Promise.resolve(
      new Response(JSON.stringify({ rows, total, page, page_size: size }), {
        status: 200,
        headers: { "content-type": "application/json" },
      })
    );
  });
}

function listUrls(spy: ReturnType<typeof mockRuns>): URL[] {
  return spy.mock.calls
    .map(([input]) => new URL(String(input), "http://localhost"))
    .filter((u) => u.pathname === "/api/evals");
}

function LocationProbe() {
  const loc = useLocation();
  return <div data-testid="location">{loc.search}</div>;
}

function renderPage(route = "/evals") {
  return renderWithProviders(
    <>
      <EvalRunsPage />
      <LocationProbe />
    </>,
    { route }
  );
}

const search = () => new URLSearchParams(screen.getByTestId("location").textContent ?? "");

describe("EvalRunsPage", () => {
  it("shows the cases rate, errored count, gate and branch per run", async () => {
    mockRuns(1);
    renderPage();
    const tr = await screen.findByTestId("eval-row-r0");
    expect(tr).toHaveTextContent("75%");
    expect(tr).toHaveTextContent("(3/4)");
    expect(within(tr).getByText("failed")).toHaveAttribute("data-gate", "failed");
    expect(tr).toHaveTextContent("main");
    expect(tr).toHaveTextContent("pytest");
    // Checks rate rides along as hover text.
    expect(tr.querySelector('[title="Cases 75% · Checks 80%"]')).not.toBeNull();
  });

  it("writes kind, gate and search filters to the URL and sends them to the API", async () => {
    const spy = mockRuns(3);
    renderPage();
    await screen.findByTestId("eval-row-r0");

    fireEvent.click(screen.getByRole("button", { name: "Evals" }));
    await waitFor(() => expect(search().get("kind")).toBe("evaluate,cli"));
    await waitFor(() => expect(listUrls(spy).at(-1)!.searchParams.get("kind")).toBe("evaluate,cli"));

    fireEvent.change(screen.getByLabelText("Gate"), { target: { value: "invalid" } });
    await waitFor(() => expect(search().get("gate")).toBe("invalid"));
    await waitFor(() => expect(listUrls(spy).at(-1)!.searchParams.get("gate")).toBe("invalid"));

    fireEvent.change(screen.getByLabelText("Search runs"), { target: { value: "nightly" } });
    await waitFor(() => expect(search().get("q")).toBe("nightly"));
    await waitFor(() => expect(listUrls(spy).at(-1)!.searchParams.get("q")).toBe("nightly"));

    fireEvent.click(screen.getByRole("button", { name: "Dataset checks" }));
    await waitFor(() => expect(search().get("kind")).toBe("ui-echo"));
  });

  it("reads filters from the URL on load", async () => {
    const spy = mockRuns(1);
    renderPage("/evals?kind=optimize&gate=passed&q=abc&page=2");
    await waitFor(() => expect(listUrls(spy).length).toBeGreaterThan(0));
    const first = listUrls(spy)[0].searchParams;
    expect(first.get("kind")).toBe("optimize");
    expect(first.get("gate")).toBe("passed");
    expect(first.get("q")).toBe("abc");
    expect(first.get("page")).toBe("2");
    expect(screen.getByRole("button", { name: "AutoLLM" })).toHaveAttribute("aria-pressed", "true");
  });

  it("filters on failed or invalid gates with one comma value", async () => {
    const spy = mockRuns(1);
    renderPage();
    await screen.findByTestId("eval-row-r0");
    fireEvent.change(screen.getByLabelText("Gate"), { target: { value: "failed,invalid" } });
    await waitFor(() => expect(search().get("gate")).toBe("failed,invalid"));
    await waitFor(() =>
      expect(listUrls(spy).at(-1)!.searchParams.get("gate")).toBe("failed,invalid")
    );
  });

  it("reads a comma gate from the URL, cleaned and in order (the Home link)", async () => {
    const spy = mockRuns(1);
    renderPage("/evals?gate=invalid,%20failed,bogus");
    await waitFor(() => expect(listUrls(spy).length).toBeGreaterThan(0));
    expect(listUrls(spy)[0].searchParams.get("gate")).toBe("failed,invalid");
    expect(screen.getByLabelText("Gate")).toHaveValue("failed,invalid");
  });

  it("keeps an unlisted gate combination selectable", async () => {
    const spy = mockRuns(1);
    renderPage("/evals?gate=passed,none");
    await waitFor(() => expect(listUrls(spy).length).toBeGreaterThan(0));
    expect(listUrls(spy)[0].searchParams.get("gate")).toBe("passed,none");
    expect(screen.getByLabelText("Gate")).toHaveValue("passed,none");
  });

  it("includes dataset checks and AutoLLM runs in the trend on request, in the URL", async () => {
    const spy = mockRuns(1);
    renderPage();
    await screen.findByTestId("eval-row-r0");
    const trendUrls = () =>
      spy.mock.calls
        .map(([input]) => new URL(String(input), "http://localhost"))
        .filter((u) => u.pathname === "/api/evals/trend");
    expect(trendUrls()[0].searchParams.has("include")).toBe(false);

    fireEvent.click(
      screen.getByRole("switch", { name: "Include dataset checks and AutoLLM runs" })
    );
    await waitFor(() => expect(search().get("trend")).toBe("all"));
    await waitFor(() => expect(trendUrls().at(-1)!.searchParams.get("include")).toBe("all"));
  });

  it("reads the trend toggle from the URL", async () => {
    const spy = mockRuns(1);
    renderPage("/evals?trend=all");
    await waitFor(() =>
      expect(
        spy.mock.calls.some(([input]) => String(input).includes("/evals/trend?include=all"))
      ).toBe(true)
    );
    expect(
      screen.getByRole("switch", { name: "Include dataset checks and AutoLLM runs" })
    ).toHaveAttribute("aria-checked", "true");
  });

  it("pages with Next and Prev, keeping the page in the URL", async () => {
    const spy = mockRuns(120);
    renderPage();
    await screen.findByTestId("eval-row-r0");
    expect(screen.getByTestId("eval-pager-range")).toHaveTextContent("1–50 of 120");

    fireEvent.click(screen.getByRole("button", { name: /Next/ }));
    await screen.findByTestId("eval-row-r50");
    expect(search().get("page")).toBe("2");
    expect(listUrls(spy).at(-1)!.searchParams.get("page")).toBe("2");
    expect(screen.getByTestId("eval-pager-range")).toHaveTextContent("51–100 of 120");

    // A filter change goes back to page 1.
    fireEvent.click(screen.getByRole("button", { name: "Pytest" }));
    await waitFor(() => expect(search().get("page")).toBeNull());
    expect(search().get("kind")).toBe("pytest");
  });
});
