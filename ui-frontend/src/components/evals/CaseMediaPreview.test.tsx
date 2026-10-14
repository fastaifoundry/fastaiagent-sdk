import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { CaseMediaPreview } from "./CaseMediaPreview";

const IMG = {
  type: "image",
  media_type: "image/png",
  size_bytes: 12_288,
  sha256: "abc123def456",
  media_idx: 1,
};

describe("CaseMediaPreview", () => {
  it("loads the stored thumbnail for an image descriptor", () => {
    render(<CaseMediaPreview value={["describe this", IMG]} runId="r1" caseId="c1" />);
    const img = screen.getByRole("img");
    expect(img).toHaveAttribute("src", "/api/evals/r1/cases/c1/media/1");
    expect(screen.getByText("describe this")).toBeInTheDocument();
    expect(screen.getByTestId("case-media-preview")).not.toHaveTextContent("media_idx");
  });

  it("falls back to a one-line description when there is no thumbnail", () => {
    render(<CaseMediaPreview value={IMG} runId="r1" caseId="c1" />);
    fireEvent.error(screen.getByRole("img"));
    expect(screen.queryByRole("img")).toBeNull();
    expect(screen.getByText("image · 12 KB · sha256 abc123…")).toBeInTheDocument();
  });

  it("describes raw bytes by size", () => {
    render(<CaseMediaPreview value={{ type: "bytes", size_bytes: 2048 }} runId="r1" caseId="c1" />);
    expect(screen.getByText("bytes · 2 KB")).toBeInTheDocument();
  });
});
