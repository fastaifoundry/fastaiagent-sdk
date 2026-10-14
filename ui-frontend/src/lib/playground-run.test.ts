/**
 * The run snapshot helpers (1.89.0, P6/P10d/P10e). `unfilledVariables` must
 * agree with the server's `_unfilled_variables`, which the response echoes.
 */
import { describe, expect, it } from "vitest";
import {
  MAX_KEPT_IMAGE_BYTES,
  saveAsEvalBody,
  snapshotImage,
  unfilledVariables,
  type PlaygroundHistoryEntry,
} from "./playground-run";

describe("unfilledVariables", () => {
  it("lists placeholders left in the text and ones given as blank", () => {
    expect(unfilledVariables("Hi {{name}}, {{topic}} {{x}}", { name: "Ada", topic: "  " })).toEqual([
      "topic",
      "x",
    ]);
  });

  it("ignores a blank value for a variable the template doesn't use", () => {
    expect(unfilledVariables("Hi {{name}}", { name: "Ada", stale: "" })).toEqual([]);
  });

  it("leaves {{@fragment}} references alone", () => {
    expect(unfilledVariables("{{@tone}} ok", {})).toEqual([]);
  });
});

describe("snapshotImage", () => {
  it("keeps a small image and drops the bytes of one over 5 MB", () => {
    expect(snapshotImage("aGVsbG8=", "image/png", "a.png")).toEqual({
      name: "a.png",
      media_type: "image/png",
      b64: "aGVsbG8=",
    });
    const big = "A".repeat(Math.ceil((MAX_KEPT_IMAGE_BYTES * 4) / 3) + 8);
    expect(snapshotImage(big, "image/png", "big.png")).toEqual({
      name: "big.png",
      media_type: "image/png",
      b64: null,
    });
    expect(snapshotImage(null, null, null)).toBeNull();
  });
});

describe("saveAsEvalBody", () => {
  const run: PlaygroundHistoryEntry = {
    id: "1",
    timestamp: "2026-10-14T00:00:00Z",
    response: "4",
    request: {
      provider: "openai",
      model: "gpt-4o-mini",
      prompt_template: "What is {{q}}?",
      variables: { q: "2+2" },
      system_prompt: "Be brief.",
      parameters: { temperature: 0.2, top_p: null, max_tokens: 64 },
      base_url: null,
      prompt_ref: { slug: "math", version: 2 },
      resolved_prompt: "What is 2+2?",
      image: { name: "a.png", media_type: "image/png", b64: "aGVsbG8=" },
    },
    metadata: {
      model: "gpt-4o-mini",
      provider: "openai",
      latency_ms: 1,
      tokens: { input: 1, output: 1 },
      cost_usd: null,
      trace_id: "abc",
      prompt: { slug: "math", version: 2, attributed: true, modified: false },
    },
  };

  it("sends the image, parameters, prompt and trace of the run", () => {
    expect(saveAsEvalBody(run, "ds", "4")).toEqual({
      dataset_name: "ds",
      input: "What is 2+2?",
      expected_output: "4",
      model: "gpt-4o-mini",
      provider: "openai",
      parameters: { temperature: 0.2, top_p: null, max_tokens: 64 },
      system_prompt: "Be brief.",
      image_b64: "aGVsbG8=",
      image_media_type: "image/png",
      image_name: "a.png",
      prompt: { slug: "math", version: 2 },
      trace_id: "abc",
    });
  });

  it("claims no prompt version for an edited template", () => {
    const edited = {
      ...run,
      metadata: { ...run.metadata, prompt: { slug: "math", version: 2, attributed: false, modified: true } },
    };
    expect(saveAsEvalBody(edited, "ds", "4")).not.toHaveProperty("prompt");
  });
});
