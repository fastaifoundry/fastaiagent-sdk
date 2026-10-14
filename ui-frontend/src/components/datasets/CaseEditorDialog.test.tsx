import { describe, expect, it, vi } from "vitest";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { CaseEditorDialog } from "./CaseEditorDialog";
import type { DatasetCase } from "@/lib/types";

/**
 * A row's fields the editor does not model (trajectory, context, a replay's
 * source trace) are shown read-only and never sent back, so a save keeps them
 * (1.89.0). The api.ts boundary (fetch) is the only mock surface.
 */

const CASE: DatasetCase = {
  index: 2,
  input: "Refund order 42",
  expected_output: "Refunded",
  tags: ["refund"],
  metadata: {},
  extra: {
    expected_trajectory: ["lookup_order", "issue_refund"],
    source_trace_id: "t-123",
  },
};

describe("CaseEditorDialog", () => {
  it("shows the row's other fields read-only and does not send them on save", async () => {
    const puts: Record<string, unknown>[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      if ((init?.method ?? "GET").toUpperCase() === "PUT" && url.endsWith("/cases/2")) {
        puts.push(JSON.parse(String(init?.body)));
      }
      return new Response(JSON.stringify({ ...CASE }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    });
    const onOpenChange = vi.fn();
    renderWithProviders(
      <CaseEditorDialog datasetName="qa" open onOpenChange={onOpenChange} caseToEdit={CASE} />
    );

    const other = await screen.findByTestId("case-other-fields");
    expect(within(other).getByText("Other fields")).toBeInTheDocument();
    expect(within(other).getByText("expected_trajectory")).toBeInTheDocument();
    expect(within(other).getByText('["lookup_order","issue_refund"]')).toBeInTheDocument();
    expect(within(other).getByText("source_trace_id")).toBeInTheDocument();
    expect(within(other).getByText('"t-123"')).toBeInTheDocument();
    // Read-only: nothing to type into.
    expect(within(other).queryByRole("textbox")).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Input"), { target: { value: "Refund order 43" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(puts).toHaveLength(1));
    expect(puts[0]!.input).toBe("Refund order 43");
    expect(puts[0]).not.toHaveProperty("extra");
    await waitFor(() => expect(onOpenChange).toHaveBeenCalledWith(false));
  });

  it("has no Other fields section when the row has none", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}"));
    renderWithProviders(
      <CaseEditorDialog
        datasetName="qa"
        open
        onOpenChange={() => {}}
        caseToEdit={{ ...CASE, extra: {} }}
      />
    );
    expect(await screen.findByText("Edit case")).toBeInTheDocument();
    expect(screen.queryByTestId("case-other-fields")).not.toBeInTheDocument();
  });
});
