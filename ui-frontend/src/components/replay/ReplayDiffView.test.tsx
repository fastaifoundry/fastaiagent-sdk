import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "@/test/utils";
import { ReplayDiffView } from "./ReplayDiffView";
import type { ComparisonResult, ReplayStep } from "@/lib/types";

function step(
  index: number,
  name: string,
  input: Record<string, unknown>,
  output: Record<string, unknown>
): ReplayStep {
  return {
    step: index,
    span_name: name,
    span_id: `span-${index}`,
    input,
    output,
    attributes: {},
    timestamp: new Date().toISOString(),
  };
}

describe("ReplayDiffView", () => {
  const original: ReplayStep[] = [
    step(0, "agent.support-bot", {}, {}),
    step(
      1,
      "llm.chat",
      { prompt: "You are a vague support bot" },
      { text: "Refunds take 14 days." }
    ),
  ];
  const rerun: ReplayStep[] = [
    step(0, "agent.support-bot", {}, {}),
    step(
      1,
      "llm.chat",
      { prompt: "Refund policy is 7 days. Be concise." },
      { text: "Refunds take 7 business days." }
    ),
  ];
  const comparison: ComparisonResult = {
    original_steps: original,
    new_steps: rerun,
    diverged_at: 1,
  };

  it("renders both side-by-side output cards", () => {
    renderWithProviders(
      <ReplayDiffView
        comparison={comparison}
        originalOutput={{ text: "14 days" }}
        newOutput={{ text: "7 business days" }}
      />
    );
    expect(screen.getByText("Original output")).toBeInTheDocument();
    expect(screen.getByText("Rerun output")).toBeInTheDocument();
  });

  it("shows the diverged-at badge", () => {
    renderWithProviders(
      <ReplayDiffView
        comparison={comparison}
        originalOutput={null}
        newOutput={null}
      />
    );
    expect(screen.getByText(/diverged at step 1/i)).toBeInTheDocument();
  });

  it("renders one step row per max(original, rerun)", () => {
    renderWithProviders(
      <ReplayDiffView
        comparison={comparison}
        originalOutput={null}
        newOutput={null}
      />
    );
    // Each row has the step index as visible text — check we see both.
    const indices = screen.getAllByText(/^[01]$/, { selector: "div" });
    expect(indices.length).toBeGreaterThanOrEqual(2);
  });

  it("expands a diverged row to show per-step input/output diff", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ReplayDiffView
        comparison={comparison}
        originalOutput={null}
        newOutput={null}
      />
    );

    // Step 1 is the diverged row with an input+output diff. The expand
    // button has an aria-label we can target.
    const expandBtn = screen.getByRole("button", {
      name: /expand step diff/i,
    });
    await user.click(expandBtn);

    // Once expanded, the per-step section labels "Input" and "Output"
    // should appear (from the DiffBlock header).
    expect(screen.getByText("Input")).toBeInTheDocument();
    expect(screen.getByText("Output")).toBeInTheDocument();
  });

  it("renders tool_diffs with both statuses, results and the disposition (1.89.0)", () => {
    const withTools: ComparisonResult = {
      original_steps: original,
      new_steps: original,
      diverged_at: null,
      tools_diverged_at: 1,
      tool_diffs: [
        {
          original_step: 1,
          new_step: 1,
          name: "lookup_order",
          original_status: "ok",
          new_status: "error",
          original_result: { status: "shipped" },
          new_result: "blocked: a rerun runs only tools you supply",
          disposition: "blocked",
        },
      ],
    };
    renderWithProviders(
      <ReplayDiffView comparison={withTools} originalOutput={null} newOutput={null} />
    );
    // The model's answers match, but the tool call did not.
    expect(screen.queryByText(/diverged at step/i)).not.toBeInTheDocument();
    expect(screen.getByText("tools differ at step 1")).toBeInTheDocument();

    const diff = screen.getByTestId("tool-diff-lookup_order");
    expect(diff).toHaveTextContent("step 1 → 1");
    expect(diff).toHaveTextContent("blocked");
    expect(diff).toHaveTextContent("Original · ok");
    expect(diff).toHaveTextContent("Rerun · error");
    expect(diff).toHaveTextContent('"status": "shipped"');
    expect(diff).toHaveTextContent("blocked: a rerun runs only tools you supply");
    expect(screen.getByText(/First tool call that changed/)).toBeInTheDocument();
  });

  it("uses the server's alignment when present, skipping memory steps", () => {
    const withMemory: ReplayStep[] = [
      step(0, "agent.support-bot", {}, {}),
      step(1, "memory.load", {}, {}),
      step(2, "llm.chat", { prompt: "p" }, { text: "Refunds take 14 days." }),
    ];
    const aligned: ComparisonResult = {
      original_steps: withMemory,
      new_steps: rerun,
      diverged_at: 2,
      new_diverged_at: 1,
      skipped_steps: [1],
      new_skipped_steps: [],
      aligned: [
        [0, 0],
        [2, 1],
      ],
    };
    renderWithProviders(
      <ReplayDiffView comparison={aligned} originalOutput={null} newOutput={null} />
    );
    // memory.load is not paired against the rerun's llm.chat.
    expect(screen.queryByText("memory.load")).not.toBeInTheDocument();
    expect(screen.getByText(/1 memory\/nested step not compared/)).toBeInTheDocument();
    expect(screen.getAllByText("llm.chat")).toHaveLength(2);
  });

  it("handles identical runs without crashing", () => {
    const identical: ComparisonResult = {
      original_steps: original,
      new_steps: original,
      diverged_at: null,
    };
    renderWithProviders(
      <ReplayDiffView
        comparison={identical}
        originalOutput={null}
        newOutput={null}
      />
    );
    // No diverged badge when diverged_at is null.
    expect(screen.queryByText(/diverged at step/i)).not.toBeInTheDocument();
  });
});
