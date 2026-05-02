/**
 * Vitest spec for ExportTraceDialog.
 *
 * Verifies the dialog opens, the checkboxes drive the URL query
 * params on the Download anchor, and the trigger button is present.
 */
import { describe, expect, it } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ExportTraceDialog } from "./ExportTraceDialog";

const TRACE_ID = "trace-abc";

function openDialog() {
  fireEvent.click(screen.getByTestId("export-trace-button"));
}

describe("ExportTraceDialog", () => {
  it("renders the trigger button", () => {
    render(<ExportTraceDialog traceId={TRACE_ID} />);
    expect(screen.getByTestId("export-trace-button")).toBeInTheDocument();
  });

  it("download URL has no query params by default", () => {
    render(<ExportTraceDialog traceId={TRACE_ID} />);
    openDialog();
    const anchor = screen.getByTestId("export-trace-download");
    expect(anchor.getAttribute("href")).toBe(
      `/api/traces/${TRACE_ID}/export`
    );
  });

  it("ticking attachments adds include_attachments=true", () => {
    render(<ExportTraceDialog traceId={TRACE_ID} />);
    openDialog();
    fireEvent.click(screen.getByTestId("export-include-attachments"));
    const anchor = screen.getByTestId("export-trace-download");
    expect(anchor.getAttribute("href")).toContain(
      "include_attachments=true"
    );
  });

  it("ticking checkpoint state adds include_checkpoint_state=true", () => {
    render(<ExportTraceDialog traceId={TRACE_ID} />);
    openDialog();
    fireEvent.click(screen.getByTestId("export-include-checkpoint-state"));
    const anchor = screen.getByTestId("export-trace-download");
    expect(anchor.getAttribute("href")).toContain(
      "include_checkpoint_state=true"
    );
  });
});
