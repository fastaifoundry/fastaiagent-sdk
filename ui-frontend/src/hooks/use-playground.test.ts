/**
 * Unit tests for the Playground SSE parser.
 *
 * `parseSseMessage` had no coverage at all before 1.53.0, which is how it
 * silently dropped the `correlation_id` on error events: the server redacts
 * provider errors to "LLM call failed." and logs the real text under that id,
 * so losing it left the user with nothing to go on. That is a large part of
 * why the Anthropic outage went unreported.
 */
import { describe, expect, it } from "vitest";
import { __parseSseMessageForTests as parseSseMessage } from "./use-playground";

describe("parseSseMessage", () => {
  it("parses a token event", () => {
    expect(parseSseMessage('event: token\ndata: {"text":"hel"}')).toEqual({
      event: "token",
      text: "hel",
    });
  });

  it("coerces a missing token text to an empty string", () => {
    expect(parseSseMessage("event: token\ndata: {}")).toEqual({
      event: "token",
      text: "",
    });
  });

  it("parses a done event with its metadata", () => {
    const raw =
      'event: done\ndata: {"metadata":{"model":"claude-opus-5",' +
      '"provider":"anthropic","latency_ms":12,"tokens":{"input":1,"output":2},' +
      '"cost_usd":0.5,"trace_id":"abc"}}';
    const parsed = parseSseMessage(raw);
    expect(parsed).toMatchObject({ event: "done" });
    expect(parsed && "metadata" in parsed && parsed.metadata).toMatchObject({
      model: "claude-opus-5",
      provider: "anthropic",
      trace_id: "abc",
    });
  });

  it("keeps the correlation id on an error event", () => {
    const raw =
      'event: error\ndata: {"message":"LLM call failed.",' +
      '"correlation_id":"4f1ff7880fcf47c4"}';
    expect(parseSseMessage(raw)).toEqual({
      event: "error",
      message: "LLM call failed.",
      correlation_id: "4f1ff7880fcf47c4",
    });
  });

  it("omits correlation_id when the server didn't send one", () => {
    const parsed = parseSseMessage('event: error\ndata: {"message":"boom"}');
    expect(parsed).toEqual({ event: "error", message: "boom" });
    expect(parsed && "correlation_id" in parsed).toBe(false);
  });

  it("falls back to a generic message when the payload has none", () => {
    expect(parseSseMessage("event: error\ndata: {}")).toEqual({
      event: "error",
      message: "stream error",
    });
  });

  it("joins multi-line data payloads", () => {
    expect(parseSseMessage('event: token\ndata: {"text":\ndata: "hi"}')).toEqual({
      event: "token",
      text: "hi",
    });
  });

  it("returns null for a comment-only or dataless frame", () => {
    expect(parseSseMessage(": keep-alive")).toBeNull();
    expect(parseSseMessage("event: token")).toBeNull();
  });

  it("returns null for malformed JSON rather than throwing", () => {
    expect(parseSseMessage("event: token\ndata: {not json")).toBeNull();
  });

  it("returns null for an unrecognised event type", () => {
    expect(parseSseMessage('event: surprise\ndata: {"a":1}')).toBeNull();
  });
});
