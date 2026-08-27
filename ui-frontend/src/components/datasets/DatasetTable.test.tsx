import { describe, expect, it, vi, beforeEach } from "vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { DatasetTable } from "./DatasetTable";
import type { DatasetSummary } from "@/lib/types";

const navigate = vi.fn();
vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  );
  return { ...actual, useNavigate: () => navigate };
});

vi.mock("@/hooks/use-datasets", () => ({
  useDeleteDataset: () => ({ mutateAsync: vi.fn(), isPending: false }),
}));

const ROWS: DatasetSummary[] = [
  {
    name: "support-cases",
    case_count: 12,
    has_multimodal: false,
    modified_at: "2026-08-27T21:44:53+00:00",
    created_at: "2026-08-20T09:00:00+00:00",
  } as DatasetSummary,
];

describe("DatasetTable", () => {
  beforeEach(() => navigate.mockClear());

  it("opens the detail page from anywhere in the row, not just the name", () => {
    renderWithProviders(<DatasetTable rows={ROWS} />);

    // Click the case-count cell — not the name.
    fireEvent.click(screen.getByText("12"));

    expect(navigate).toHaveBeenCalledWith("/datasets/support-cases");
  });

  it("keeps the name a real anchor so it can open in a new tab", () => {
    renderWithProviders(<DatasetTable rows={ROWS} />);
    expect(screen.getByRole("link", { name: "support-cases" })).toHaveAttribute(
      "href",
      "/datasets/support-cases"
    );
  });

  it("does not double-navigate when the name itself is clicked", async () => {
    renderWithProviders(<DatasetTable rows={ROWS} />);
    fireEvent.click(screen.getByRole("link", { name: "support-cases" }));
    await waitFor(() => expect(navigate).not.toHaveBeenCalled());
  });

  it("does not navigate when the delete action is used", async () => {
    // The row is clickable now, so a destructive action inside it must not
    // also fire the row handler — otherwise confirming a delete would yank
    // the user onto the detail page of the thing they just removed.
    renderWithProviders(<DatasetTable rows={ROWS} />);
    fireEvent.click(screen.getByTitle("Delete dataset"));

    expect(await screen.findByText(/Delete dataset 'support-cases'\?/)).toBeInTheDocument();
    expect(navigate).not.toHaveBeenCalled();
  });

  it("does not navigate when the export action is used", () => {
    renderWithProviders(<DatasetTable rows={ROWS} />);
    fireEvent.click(screen.getByTitle("Export as JSONL"));
    expect(navigate).not.toHaveBeenCalled();
  });
});
