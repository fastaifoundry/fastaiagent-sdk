/**
 * What a failed Playground run shows (1.89.0, P8).
 *
 * Every provider failure used to read "LLM call failed." Now the server sends
 * a kind, the provider's (scrubbed) message and a hint; in redacted mode it
 * sends only the opaque message and a correlation id.
 */
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { RunErrorPanel } from "./RunErrorPanel";
import { normalizeErrorBody } from "@/lib/api-error";

describe("RunErrorPanel", () => {
  it("shows the kind, the provider's message, the hint and the correlation id", () => {
    const info = normalizeErrorBody(
      {
        detail: {
          error: "LLM call failed.",
          correlation_id: "abc123",
          kind: "bad_request",
          provider_status: 400,
          provider_message: "`temperature` is deprecated for this model.",
          hint: "This Claude model accepts only its default temperature (1).",
        },
      },
      502
    );
    render(<RunErrorPanel info={info} />);
    expect(screen.getByTestId("run-error-kind")).toHaveTextContent(
      "Request rejected by the provider (HTTP 400)"
    );
    expect(screen.getByText("`temperature` is deprecated for this model.")).toBeInTheDocument();
    expect(screen.getByText(/only its default temperature/)).toBeInTheDocument();
    expect(screen.getByText("abc123")).toBeInTheDocument();
    expect(screen.queryByText(/not shown here/)).toBeNull();
  });

  it("falls back to the opaque message in redacted mode", () => {
    const info = normalizeErrorBody(
      { detail: { error: "LLM call failed.", correlation_id: "abc123" } },
      502
    );
    render(<RunErrorPanel info={info} />);
    expect(screen.queryByTestId("run-error-kind")).toBeNull();
    expect(screen.getByText("LLM call failed.")).toBeInTheDocument();
    expect(screen.getByText(/not shown here/)).toHaveTextContent("abc123");
  });

  it("shows a connection failure's hint without a provider message", () => {
    render(
      <RunErrorPanel
        info={{
          message: "LLM call failed.",
          fieldErrors: [],
          correlationId: "c1",
          kind: "connection",
          hint: "Could not connect to http://localhost:11434.",
        }}
      />
    );
    expect(screen.getByTestId("run-error-kind")).toHaveTextContent("Could not connect");
    expect(screen.getByText(/localhost:11434/)).toBeInTheDocument();
    expect(screen.queryByText(/not shown here/)).toBeNull();
  });

  it("renders a config 400's string detail", () => {
    const info = normalizeErrorBody(
      { detail: "Provider 'custom' needs an Endpoint (base URL): set it in the Connection panel." },
      400
    );
    render(<RunErrorPanel info={info} />);
    expect(screen.getByText(/needs an Endpoint/)).toBeInTheDocument();
  });
});
