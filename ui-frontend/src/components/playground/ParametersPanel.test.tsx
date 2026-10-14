/**
 * Sampling-rule hints on the parameter controls (1.89.0, P7).
 *
 * Claude 5 accepts only its default temperature and no top_p; older Claude
 * accepts one of the two. The panel shows it before the call instead of a
 * bare provider 400 after it. Real component, no mocks.
 */
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ParametersPanel } from "./ParametersPanel";
import { FREE_RULE } from "@/lib/sampling-rules";
import type { PlaygroundParameters, SamplingRule } from "@/lib/types";

const CLAUDE_NEW: SamplingRule = {
  ...FREE_RULE,
  temperature: "default_only",
  top_p: "unsupported",
  note: "This Claude model accepts only its default temperature (1) and no top_p.",
};
const CLAUDE_OLD: SamplingRule = {
  ...FREE_RULE,
  exclusive: true,
  note: "Claude accepts temperature or top_p, not both.",
};

function Host({ rule, initial }: { rule: SamplingRule; initial?: Partial<PlaygroundParameters> }) {
  const [value, setValue] = useState<PlaygroundParameters>({
    temperature: null,
    top_p: null,
    max_tokens: 1024,
    ...initial,
  });
  return (
    <>
      <ParametersPanel value={value} onChange={setValue} rule={rule} />
      <pre data-testid="state">{JSON.stringify(value)}</pre>
    </>
  );
}

const state = () => JSON.parse(screen.getByTestId("state").textContent ?? "{}");

describe("ParametersPanel sampling rules", () => {
  it("disables top_p and pins temperature for a Claude 5 model", () => {
    render(<Host rule={CLAUDE_NEW} initial={{ temperature: 0.4 }} />);
    expect(screen.getByTestId("sampling-rule-note")).toHaveTextContent("no top_p");
    expect(screen.getByRole("switch", { name: "Send top_p" })).toBeDisabled();
    expect(screen.getByTestId("param-top-p-rule")).toHaveTextContent("does not accept top_p");
    // Temperature can still be sent, but only at the default.
    expect(screen.getByRole("switch", { name: "Send temperature" })).toBeEnabled();
    expect(screen.getByTestId("param-temperature-rule")).toHaveTextContent("only the default");
    expect(screen.getByText("1.00")).toBeInTheDocument();
  });

  it("switching one on switches the other off under an exclusive rule", () => {
    render(<Host rule={CLAUDE_OLD} initial={{ temperature: 0.7 }} />);
    fireEvent.click(screen.getByRole("switch", { name: "Send top_p" }));
    expect(state()).toMatchObject({ top_p: 1, temperature: null });
  });

  it("warns when both are already on under an exclusive rule", () => {
    render(<Host rule={CLAUDE_OLD} initial={{ temperature: 0.7, top_p: 0.9 }} />);
    expect(screen.getByRole("alert")).toHaveTextContent("not both");
  });

  it("says nothing for a model without limits", () => {
    render(<Host rule={FREE_RULE} />);
    expect(screen.queryByTestId("sampling-rule-note")).toBeNull();
    expect(screen.getByRole("switch", { name: "Send top_p" })).toBeEnabled();
  });

  it("still flags a max_tokens out of range", () => {
    render(<Host rule={FREE_RULE} />);
    fireEvent.change(screen.getByLabelText("max_tokens"), { target: { value: "300000" } });
    expect(screen.getByRole("alert")).toHaveTextContent("max_tokens must be");
  });
});
