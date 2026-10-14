import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { PlaygroundTraceLink } from "./PlaygroundTraceLink";

/**
 * A Playground trace written to a store this UI doesn't read used to get a
 * link that 404'd. ``trace_in_ui_db: false`` now renders a note instead
 * (1.89.0).
 */
describe("PlaygroundTraceLink", () => {
  it("links to the trace when it is in the UI's database", () => {
    renderWithProviders(<PlaygroundTraceLink traceId="abc123" inUiDb />);
    expect(screen.getByRole("link", { name: /trace/ })).toHaveAttribute(
      "href",
      "/traces/abc123"
    );
  });

  it("links when an older server sends no trace_in_ui_db", () => {
    renderWithProviders(<PlaygroundTraceLink traceId="abc123" />);
    expect(screen.getByRole("link", { name: /trace/ })).toBeInTheDocument();
  });

  it("is not a link when the trace went to another store", () => {
    renderWithProviders(
      <PlaygroundTraceLink traceId="abcdef0123456789" inUiDb={false} />
    );
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    const note = screen.getByTestId("playground-trace-elsewhere");
    expect(note).toHaveTextContent("trace abcdef01…6789 was written to another store");
    expect(note).toHaveAttribute("title", expect.stringContaining("abcdef0123456789"));
  });

  it("renders nothing without a trace id", () => {
    const { container } = renderWithProviders(
      <PlaygroundTraceLink traceId={null} inUiDb={false} />
    );
    expect(container).toBeEmptyDOMElement();
  });
});
