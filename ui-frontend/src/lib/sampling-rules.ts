/**
 * Which sampling parameters a model accepts (1.89.0, P7).
 *
 * A port of `fastaiagent.llm.param_rules.sampling_rule`, run over the table
 * `GET /api/playground/models` returns, so the free-text model field resolves
 * on every keystroke without a request. Keep the two in step: the cases in
 * `sampling-rules.test.ts` mirror `test_sampling_rule_resolution`.
 *
 * Hints only. The server never blocks a call on these rules (decision D4):
 * they go stale as providers change, and the provider stays the authority.
 */
import type { PlaygroundParameters, SamplingRule, SamplingRuleRow } from "@/lib/types";

export const FREE_RULE: SamplingRule = {
  temperature: "free",
  top_p: "free",
  exclusive: false,
  default_temperature: 1,
  default_top_p: 1,
  note: "",
};

/** Model-id prefixes added by routers and clouds in front of the vendor's id. */
const STRIP = [
  "us.anthropic.",
  "eu.anthropic.",
  "apac.anthropic.",
  "anthropic.",
  "anthropic/",
  "openai/",
] as const;

/** Providers whose wire format carries another vendor's models. */
const VENDOR_OF: Record<string, string | null> = {
  azure: "openai",
  openrouter: null,
  bedrock: "anthropic",
};

function vendorAndModel(provider: string, model: string): [string, string] {
  let m = (model ?? "").trim().toLowerCase();
  let vendor = provider;
  const prefix = STRIP.find((p) => m.startsWith(p));
  if (prefix) {
    vendor = prefix.includes("anthropic") ? "anthropic" : "openai";
    m = m.slice(prefix.length);
  } else if (provider in VENDOR_OF) {
    vendor = VENDOR_OF[provider] ?? provider;
  }
  return [vendor, m];
}

/** The rule for `model` on `provider`; {@link FREE_RULE} when nothing matches. */
export function resolveSamplingRule(
  rows: SamplingRuleRow[] | undefined,
  provider: string,
  model: string
): SamplingRule {
  const [vendor, m] = vendorAndModel(provider, model);
  let best: SamplingRuleRow | null = null;
  for (const row of rows ?? []) {
    if (
      row.provider === vendor &&
      m.startsWith(row.model_prefix) &&
      (best === null || row.model_prefix.length > best.model_prefix.length)
    ) {
      best = row;
    }
  }
  if (!best) return FREE_RULE;
  const { provider: _p, model_prefix: _m, ...rule } = best;
  void _p;
  void _m;
  return rule;
}

/** True when the rule limits anything, so the panel has something to say. */
export function ruleLimits(rule: SamplingRule): boolean {
  return rule.temperature !== "free" || rule.top_p !== "free" || rule.exclusive;
}

/**
 * The parameters a run actually sends under `rule`: an `unsupported` one is
 * dropped and a `default_only` one is pinned to its default. Both together
 * under an `exclusive` rule are left alone: the panel warns, and the
 * provider's error says the rest.
 */
export function applySamplingRule(
  params: PlaygroundParameters,
  rule: SamplingRule
): PlaygroundParameters {
  const pin = (
    value: number | null,
    support: SamplingRule["temperature"],
    fallback: number
  ): number | null => {
    if (value === null || support === "unsupported") return null;
    return support === "default_only" ? fallback : value;
  };
  return {
    ...params,
    temperature: pin(params.temperature, rule.temperature, rule.default_temperature),
    top_p: pin(params.top_p, rule.top_p, rule.default_top_p),
  };
}
