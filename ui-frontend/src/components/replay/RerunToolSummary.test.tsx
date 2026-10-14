import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { RerunToolSummary } from "./RerunToolSummary";
import type { RerunResult } from "@/lib/types";

/**
 * The rerun result says what happened to each tool, warns when the model
 * called a blocked one (its answer can't show that tool's effect), and says
 * the rerun stayed local (1.89.0).
 */

const BASE: RerunResult = {
  fork_id: "f1",
  new_trace_id: "t2",
  new_output: "ok",
  original_output: "ok",
  steps_executed: 3,
  source: "replay",
  determinism: "recorded",
  tool_policy: "block",
  tool_dispositions: {
    lookup_order: "live",
    issue_refund: "stub",
    send_email: "blocked",
    legacy: "no_function",
  },
  blocked_tool_calls: { send_email: 2 },
  exported: false,
};

describe("RerunToolSummary", () => {
  it("renders one disposition chip per tool", () => {
    renderWithProviders(<RerunToolSummary result={BASE} />);
    expect(screen.getByTestId("tool-disposition-lookup_order")).toHaveTextContent("lookup_order· live");
    expect(screen.getByTestId("tool-disposition-issue_refund")).toHaveTextContent("· stub");
    expect(screen.getByTestId("tool-disposition-send_email")).toHaveTextContent("· blocked");
    expect(screen.getByTestId("tool-disposition-legacy")).toHaveTextContent("· no_function");
    expect(screen.getByText("LLM: recorded · tool policy: block")).toBeInTheDocument();
  });

  it("warns that blocked calls are not reflected in the answer", () => {
    renderWithProviders(<RerunToolSummary result={BASE} />);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("send_email ×2");
    expect(alert).toHaveTextContent("the rerun's answer does not show their effect");
  });

  it("has no warning when no blocked tool was called", () => {
    renderWithProviders(
      <RerunToolSummary result={{ ...BASE, blocked_tool_calls: {} }} />
    );
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("says a non-exported rerun is local only", () => {
    renderWithProviders(<RerunToolSummary result={BASE} />);
    expect(
      screen.getByText(/Not counted in Analytics and not exported/i)
    ).toBeInTheDocument();
  });

  it("renders nothing new for an older server's rerun payload", () => {
    const legacy: RerunResult = {
      fork_id: "f1",
      new_trace_id: "t2",
      new_output: null,
      original_output: null,
      steps_executed: 1,
    };
    renderWithProviders(<RerunToolSummary result={legacy} />);
    expect(screen.getByTestId("rerun-tool-summary")).toBeEmptyDOMElement();
  });
});
