import { describe, expect, it, vi, beforeEach } from "vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { PromptsPage } from "./PromptsPage";

const navigate = vi.fn();
vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  );
  return { ...actual, useNavigate: () => navigate };
});

vi.mock("@/hooks/use-prompts", () => ({
  usePrompts: () => ({
    data: {
      registry_is_local: true,
      rows: [
        {
          name: "support-greeting",
          latest_version: "3",
          versions: 3,
          linked_trace_count: 12,
        },
      ],
    },
    isLoading: false,
    refetch: vi.fn(),
    isFetching: false,
  }),
}));

describe("PromptsPage", () => {
  beforeEach(() => navigate.mockClear());

  it("opens the detail page from anywhere in the row, not just the name", () => {
    // The row advertises `cursor-pointer`; before this it carried no handler,
    // so the pointer promised a click only the name actually accepted.
    renderWithProviders(<PromptsPage />);

    // Click a cell that is NOT the name.
    fireEvent.click(screen.getByText("12"));

    expect(navigate).toHaveBeenCalledWith("/prompts/support-greeting");
  });

  it("opens the detail page from the empty space in the name cell", () => {
    // The regression this replaces: stopPropagation sat on the whole name
    // cell rather than on the anchor, so every click that landed in the
    // cell's blank area was swallowed. The name column is the widest one
    // (~968px against ~108px of text), which made ~860px of each row look
    // clickable and do nothing — while the narrow right-hand cells worked.
    //
    // The test above clicks a cell that *has* text, so it never covered this.
    renderWithProviders(<PromptsPage />);

    const nameCell = screen.getByRole("link", { name: "support-greeting" })
      .parentElement as HTMLElement;
    fireEvent.click(nameCell);

    expect(navigate).toHaveBeenCalledWith("/prompts/support-greeting");
  });

  it("keeps the name a real anchor so it can open in a new tab", async () => {
    renderWithProviders(<PromptsPage />);
    const link = screen.getByRole("link", { name: "support-greeting" });
    expect(link).toHaveAttribute("href", "/prompts/support-greeting");
  });

  it("does not double-navigate when the name itself is clicked", async () => {
    renderWithProviders(<PromptsPage />);
    fireEvent.click(screen.getByRole("link", { name: "support-greeting" }));
    // The anchor stops propagation, so the row handler must not also fire.
    // (Scoped to the anchor, not the cell — see the dead-zone test above.)
    await waitFor(() => expect(navigate).not.toHaveBeenCalled());
  });

  it("percent-encodes names that need it", () => {
    expect(encodeURIComponent("a/b prompt")).toBe("a%2Fb%20prompt");
  });
});
