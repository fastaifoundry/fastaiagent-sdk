/**
 * The TS port of `param_rules.sampling_rule` (1.89.0, P7).
 *
 * The table below is `rules_table()` as the server returns it, and the cases
 * are `tests/test_ui_playground_truth.py::test_sampling_rule_resolution`, so a
 * drift between the two resolvers fails here.
 */
import { describe, expect, it } from "vitest";
import { applySamplingRule, FREE_RULE, resolveSamplingRule, ruleLimits } from "./sampling-rules";
import type { SamplingRule, SamplingRuleRow } from "./types";

const CLAUDE_NEW: SamplingRule = {
  temperature: "default_only",
  top_p: "unsupported",
  exclusive: false,
  default_temperature: 1,
  default_top_p: 1,
  note: "This Claude model accepts only its default temperature (1) and no top_p.",
};
const CLAUDE_OLD: SamplingRule = {
  ...FREE_RULE,
  exclusive: true,
  note: "Claude accepts temperature or top_p, not both.",
};
const OPENAI_REASONING: SamplingRule = {
  ...FREE_RULE,
  temperature: "default_only",
  top_p: "default_only",
  note: "This OpenAI reasoning model accepts only the default temperature (1) and top_p (1).",
};
const OPENAI_O: SamplingRule = {
  ...FREE_RULE,
  temperature: "default_only",
  top_p: "unsupported",
  note: "This OpenAI o-series model accepts only the default temperature (1) and no top_p.",
};

const row = (provider: string, model_prefix: string, rule: SamplingRule): SamplingRuleRow => ({
  provider,
  model_prefix,
  ...rule,
});

const RULES: SamplingRuleRow[] = [
  row("anthropic", "claude-", CLAUDE_OLD),
  row("anthropic", "claude-opus-4-7", CLAUDE_NEW),
  row("anthropic", "claude-opus-4-8", CLAUDE_NEW),
  row("anthropic", "claude-sonnet-5", CLAUDE_NEW),
  row("anthropic", "claude-opus-5", CLAUDE_NEW),
  row("anthropic", "claude-haiku-5", CLAUDE_NEW),
  row("anthropic", "claude-fable-5", CLAUDE_NEW),
  row("openai", "gpt-5", OPENAI_REASONING),
  row("openai", "gpt-5.1", FREE_RULE),
  row("openai", "gpt-5.2", FREE_RULE),
  row("openai", "o1", OPENAI_O),
  row("openai", "o3", OPENAI_O),
  row("openai", "o4", OPENAI_REASONING),
];

describe("resolveSamplingRule", () => {
  it.each([
    ["anthropic", "claude-haiku-4-5", "free", "free", true],
    ["anthropic", "claude-sonnet-5", "default_only", "unsupported", false],
    ["bedrock", "us.anthropic.claude-opus-5-v1", "default_only", "unsupported", false],
    ["openrouter", "anthropic/claude-fable-5", "default_only", "unsupported", false],
    ["openai", "gpt-5-mini", "default_only", "default_only", false],
    ["openai", "gpt-5.1", "free", "free", false],
    ["openai", "o3-mini", "default_only", "unsupported", false],
    ["azure", "gpt-5", "default_only", "default_only", false],
    ["ollama", "llama3.2", "free", "free", false],
  ] as const)("%s %s", (provider, model, temperature, top_p, exclusive) => {
    const rule = resolveSamplingRule(RULES, provider, model);
    expect([rule.temperature, rule.top_p, rule.exclusive]).toEqual([temperature, top_p, exclusive]);
  });

  it("ignores case and surrounding space in the typed model", () => {
    expect(resolveSamplingRule(RULES, "anthropic", "  Claude-Sonnet-5 ").temperature).toBe(
      "default_only"
    );
  });

  it("is free without a table (an older server)", () => {
    expect(resolveSamplingRule(undefined, "anthropic", "claude-sonnet-5")).toEqual(FREE_RULE);
    expect(ruleLimits(FREE_RULE)).toBe(false);
  });

  it("returns the rule without the row's own keys", () => {
    const rule = resolveSamplingRule(RULES, "anthropic", "claude-opus-5");
    expect(rule).toEqual(CLAUDE_NEW);
    expect(rule).not.toHaveProperty("model_prefix");
  });
});

describe("applySamplingRule", () => {
  const params = { temperature: 0.4, top_p: 0.9, max_tokens: 64 };

  it("drops an unsupported parameter and pins a default-only one", () => {
    expect(applySamplingRule(params, CLAUDE_NEW)).toEqual({
      temperature: 1,
      top_p: null,
      max_tokens: 64,
    });
  });

  it("leaves a parameter the user switched off unsent", () => {
    expect(applySamplingRule({ ...params, temperature: null }, CLAUDE_NEW).temperature).toBeNull();
  });

  it("sends free parameters as set, both under an exclusive rule (the panel warns)", () => {
    expect(applySamplingRule(params, FREE_RULE)).toEqual(params);
    expect(applySamplingRule(params, CLAUDE_OLD)).toEqual(params);
  });
});
