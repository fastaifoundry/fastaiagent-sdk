/**
 * Any 422 used to crash the Playground (1.89.0, P1).
 *
 * FastAPI's 422 `detail` is a list of objects; the UI rendered it as a React
 * child, React threw, and the error boundary replaced the page (form lost).
 * Elsewhere the same body showed as "[object Object]".
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { normalizeErrorBody } from "./api-error";
import { ApiError, api } from "./api";
import { streamPlayground } from "@/hooks/use-playground";
import { RunErrorPanel } from "@/components/playground/RunErrorPanel";

const KEY = "sk-should-never-appear-0123456789";

const LIST_422 = {
  detail: [
    {
      type: "less_than_equal",
      loc: ["body", "parameters", "max_tokens"],
      msg: "Input should be less than or equal to 200000",
      input: 999999,
      ctx: { le: 200000 },
    },
    {
      type: "string_too_long",
      loc: ["body", "api_key"],
      msg: "String should have at most 512 characters",
      input: KEY,
    },
  ],
};

afterEach(() => vi.restoreAllMocks());

function stubFetch(status: number, body: unknown) {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(
    new Response(JSON.stringify(body), {
      status,
      statusText: "Unprocessable Entity",
      headers: { "content-type": "application/json" },
    })
  );
}

describe("normalizeErrorBody", () => {
  it("turns a 422 list into field errors with a string message", () => {
    const info = normalizeErrorBody(LIST_422, 422);
    expect(typeof info.message).toBe("string");
    expect(info.fieldErrors).toEqual([
      { field: "parameters.max_tokens", message: "Input should be less than or equal to 200000" },
      { field: "api_key", message: "String should have at most 512 characters" },
    ]);
  });

  it("never reads `input`, which can echo the key", () => {
    expect(JSON.stringify(normalizeErrorBody(LIST_422, 422))).not.toContain(KEY);
  });

  it("reads an object detail (the /run 502 shape)", () => {
    const info = normalizeErrorBody(
      { detail: { error: "LLM call failed.", correlation_id: "abc" } },
      502
    );
    expect(info.message).toBe("LLM call failed.");
    expect(info.correlationId).toBe("abc");
  });

  it("keeps a string detail and falls back to the status text", () => {
    expect(normalizeErrorBody({ detail: "nope" }, 400).message).toBe("nope");
    expect(normalizeErrorBody(null, 500, "Server Error").message).toBe("Server Error");
  });
});

describe("the call sites", () => {
  it("api.ts raises an ApiError with a readable message, not [object Object]", async () => {
    stubFetch(422, LIST_422);
    const err = (await api.post("/datasets/x/run-eval", {}).catch((e) => e)) as ApiError;
    expect(err).toBeInstanceOf(ApiError);
    expect(err.message).not.toContain("[object Object]");
    expect(err.info.fieldErrors).toHaveLength(2);
  });

  it("the Playground stream yields a string message and the field errors", async () => {
    stubFetch(422, LIST_422);
    const events = [];
    for await (const ev of streamPlayground({
      provider: "custom",
      model: "m",
      prompt_template: "hi",
      variables: {},
    } as never)) {
      events.push(ev);
    }
    expect(events).toHaveLength(1);
    const ev = events[0] as { event: string; message: unknown; field_errors?: unknown[] };
    expect(ev.event).toBe("error");
    expect(typeof ev.message).toBe("string");
    expect(ev.field_errors).toHaveLength(2);
  });

  it("the error panel renders a 422 instead of crashing", () => {
    render(<RunErrorPanel info={normalizeErrorBody(LIST_422, 422)} />);
    expect(screen.getByTestId("run-error")).toHaveTextContent("max_tokens");
    expect(screen.getByTestId("run-error")).toHaveTextContent("Token");
    expect(screen.getByTestId("run-error")).not.toHaveTextContent(KEY);
  });
});
