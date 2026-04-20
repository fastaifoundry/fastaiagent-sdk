import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { WorkflowBadge } from "./WorkflowBadge";

describe("WorkflowBadge", () => {
  it("renders the lowercase runner type label", () => {
    render(<WorkflowBadge type="agent" />);
    expect(screen.getByText("agent")).toBeInTheDocument();
  });

  it("maps chain/swarm/supervisor to distinct labels", () => {
    for (const type of ["chain", "swarm", "supervisor"] as const) {
      const { unmount } = render(<WorkflowBadge type={type} />);
      expect(screen.getByText(type)).toBeInTheDocument();
      unmount();
    }
  });

  it("includes the runner name when variant=full", () => {
    render(
      <WorkflowBadge type="chain" name="support-pipeline" variant="full" />
    );
    expect(screen.getByText(/chain/)).toBeInTheDocument();
    expect(screen.getByText(/support-pipeline/)).toBeInTheDocument();
  });

  it("hides the runner name by default", () => {
    render(<WorkflowBadge type="chain" name="support-pipeline" />);
    expect(screen.queryByText(/support-pipeline/)).toBeNull();
    // But it's in the title attribute for hover tooltips.
    expect(screen.getByTitle("support-pipeline")).toBeInTheDocument();
  });
});
