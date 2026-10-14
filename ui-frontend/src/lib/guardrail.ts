/**
 * Shared guardrail vocabulary for every surface that shows a check: the span
 * tree, the span inspector, the Scores card, the events list and the event
 * detail page.
 *
 * 1.89.0 — a guardrail span's OTel status is ERROR for *any* non-pass verdict,
 * a ``warn`` that let the run finish included, so nothing here reads
 * ``span.status``. Rows say what the check did to the run (``effect``), not
 * only its verdict (``outcome``); rows written before 1.89.0 carry no effect
 * and are read the way they always were ("legacy").
 */
import type { GuardrailEffect, GuardrailEvent, SpanRow } from "@/lib/types";

export type GuardrailTone = "success" | "warning" | "danger" | "info" | "neutral";

/** Chip background + text. */
export const TONE_CHIP: Record<GuardrailTone, string> = {
  success: "bg-fa-success/10 text-fa-success",
  warning: "bg-fa-warning/10 text-fa-warning",
  danger: "bg-destructive/10 text-destructive",
  info: "bg-fa-info/10 text-fa-info",
  neutral: "bg-muted text-muted-foreground",
};

/** Bordered badge (detail page). */
export const TONE_BORDER: Record<GuardrailTone, string> = {
  success: "border-fa-success/40",
  warning: "border-fa-warning/40",
  danger: "border-destructive/40",
  info: "border-fa-info/40",
  neutral: "border-border",
};

/** Foreground only (icons). */
export const TONE_TEXT: Record<GuardrailTone, string> = {
  success: "text-fa-success",
  warning: "text-fa-warning",
  danger: "text-destructive",
  info: "text-fa-info",
  neutral: "text-muted-foreground",
};

/** Timeline bar fill (span tree). */
export const TONE_BAR: Record<GuardrailTone, string> = {
  success: "bg-fa-success/70",
  warning: "bg-fa-warning/70",
  danger: "bg-destructive/60",
  info: "bg-fa-info/70",
  neutral: "bg-muted-foreground/40",
};

export interface GuardrailVerdict {
  /** Lower case; surfaces render it upper case. */
  label: string;
  tone: GuardrailTone;
}

/** The events list filtered to one agent's rule (at one position, if given). */
export function guardrailEventsHref(
  agent: string,
  rule: string | null,
  position?: string | null,
): string {
  const params = new URLSearchParams({ agent });
  if (rule) params.set("rule", rule);
  if (position) params.set("position", position);
  return `/guardrails?${params.toString()}`;
}

// ---------------------------------------------------------------------------
// Outcome (the existing ``outcome`` column) — one table for every surface.
// ---------------------------------------------------------------------------

export const OUTCOME_META: Record<string, GuardrailVerdict> = {
  passed: { label: "passed", tone: "success" },
  blocked: { label: "blocked", tone: "danger" },
  warned: { label: "warned", tone: "warning" },
  // The payload was masked or overridden and the run continued.
  filtered: { label: "filtered", tone: "info" },
  reasked: { label: "re-asked", tone: "warning" },
  // Not a verdict: the check could not run, and on_error decided.
  errored: { label: "could not run", tone: "warning" },
};

export function outcomeMeta(outcome: string | null | undefined): GuardrailVerdict {
  const key = (outcome ?? "").toLowerCase();
  return OUTCOME_META[key] ?? { label: outcome || "unknown", tone: "neutral" };
}

// ---------------------------------------------------------------------------
// Effect — what the check did to the run.
// ---------------------------------------------------------------------------

export const EFFECT_META: Record<GuardrailEffect, GuardrailVerdict> = {
  none: { label: "pass", tone: "success" },
  warned: { label: "warned", tone: "warning" },
  rewritten: { label: "rewrote", tone: "info" },
  reasked: { label: "re-asked", tone: "warning" },
  blocked: { label: "blocked", tone: "danger" },
  observed: { label: "observed", tone: "neutral" },
  unchecked: { label: "unchecked", tone: "warning" },
};

/** Shorter wording for an effect shown next to an outcome. */
const EFFECT_SUFFIX: Record<GuardrailEffect, string> = {
  none: "no effect",
  warned: "warned",
  rewritten: "rewrote",
  reasked: "re-asked",
  blocked: "blocked",
  observed: "observe-only",
  unchecked: "unchecked",
};

/** The effect an outcome implies on its own; anything else is worth saying. */
const NATURAL_EFFECT: Record<string, GuardrailEffect> = {
  passed: "none",
  blocked: "blocked",
  warned: "warned",
  filtered: "rewritten",
  reasked: "reasked",
};

/**
 * Text to show after an outcome chip, or null when the effect adds nothing
 * (or is unknown — a legacy row). ``errored`` always says what on_error did.
 */
export function effectSuffix(
  outcome: string | null | undefined,
  effect: GuardrailEffect | string | null | undefined,
): string | null {
  if (!effect || !(effect in EFFECT_SUFFIX)) return null;
  if (NATURAL_EFFECT[(outcome ?? "").toLowerCase()] === effect) return null;
  return EFFECT_SUFFIX[effect as GuardrailEffect];
}

// ---------------------------------------------------------------------------
// Span verdict
// ---------------------------------------------------------------------------

function flag(value: unknown): boolean | null {
  if (value === true || value === 1) return true;
  if (value === false || value === 0) return false;
  if (typeof value === "string") {
    const v = value.trim().toLowerCase();
    if (v === "true" || v === "1") return true;
    if (v === "false" || v === "0") return false;
  }
  return null;
}

export function isGuardrailSpan(span: Pick<SpanRow, "name">): boolean {
  return span.name.startsWith("guardrail.");
}

/**
 * What a guardrail span decided, or null for any other span (and for a
 * guardrail span that carries nothing to read — the caller falls back to the
 * span status).
 *
 * Prefers the local event row's ``effect`` (``span.guardrail``); otherwise
 * derives the verdict from the span's own attributes.
 */
export function guardrailSpanVerdict(
  span: Pick<SpanRow, "name" | "attributes" | "guardrail">,
): GuardrailVerdict | null {
  const effect = span.guardrail?.effect;
  if (effect && effect in EFFECT_META) return EFFECT_META[effect];
  if (!isGuardrailSpan(span) && !span.guardrail) return null;

  const attrs = span.attributes ?? {};
  const passed = flag(attrs["fastaiagent.guardrail.passed"]);
  const errored = flag(attrs["fastaiagent.guardrail.errored"]);
  const taken = attrs["fastaiagent.guardrail.action_taken"];

  if (errored) {
    return passed
      ? { label: "unchecked", tone: "warning" }
      : { label: "could not run", tone: "warning" };
  }
  if (passed === true) return { label: "pass", tone: "success" };
  switch (taken) {
    case "warned":
      return { label: "warned", tone: "warning" };
    case "masked":
    case "overridden":
      return { label: "rewrote", tone: "info" };
    case "reask":
      return { label: "re-asked", tone: "warning" };
    case "blocked":
      return { label: "blocked", tone: "danger" };
  }
  // A failed check from before actions were recorded (pre-1.57.0).
  if (passed === false) return { label: "failed", tone: "danger" };
  return null;
}

// ---------------------------------------------------------------------------
// Action chain: action → action_taken → effect
// ---------------------------------------------------------------------------

/** What each configured action produces when it gets what it asked for. */
export const ACTION_FULFILLED: Record<string, string> = {
  block: "blocked",
  warn: "warned",
  mask: "masked",
  override: "overridden",
  reask: "reask",
};

/** The effect each resolved action has on a blocking rule. */
const TAKEN_EFFECT: Record<string, GuardrailEffect> = {
  none: "none",
  blocked: "blocked",
  warned: "warned",
  masked: "rewritten",
  overridden: "rewritten",
  reask: "reasked",
};

type ActionFields = Pick<
  GuardrailEvent,
  "action" | "action_taken" | "effect" | "final_reason" | "outcome"
>;

/**
 * True when the configured action, the action taken and the effect do not
 * line up — an errored check that blocked, a mask with nothing to mask, a
 * rewrite the caller refused, an observe-only rule. Worth a ``*``.
 */
export function actionDiverged(ev: ActionFields): boolean {
  const taken = ev.action_taken;
  if (!taken || taken === "none") return false;
  if (ev.action && ACTION_FULFILLED[ev.action] && ACTION_FULFILLED[ev.action] !== taken) {
    return true;
  }
  if (ev.effect && ev.effect !== "observed" && TAKEN_EFFECT[taken] && TAKEN_EFFECT[taken] !== ev.effect) {
    return true;
  }
  return false;
}

/**
 * The events list's Action cell: ``masked · observe-only``,
 * ``blocked (was masked)``, or the action taken. Null when nothing was done.
 */
export function actionSummary(
  ev: ActionFields,
): { text: string; diverged: boolean; title?: string } | null {
  const taken = ev.action_taken;
  if (!taken || taken === "none") return null;
  if (ev.effect === "observed") {
    return {
      text: `${taken} · observe-only`,
      diverged: false,
      title: "Observe-only rule: recorded what it would have done; nothing was applied",
    };
  }
  if (ev.effect === "blocked" && taken !== "blocked") {
    return {
      text: `blocked (was ${taken})`,
      diverged: true,
      title: ev.final_reason ?? `The ${taken} could not be applied, so the run was blocked`,
    };
  }
  const diverged = actionDiverged(ev);
  return {
    text: taken,
    diverged,
    title: diverged
      ? (ev.final_reason ??
        `configured as "${ev.action}", but the action could not be carried out`)
      : (ev.final_reason ?? undefined),
  };
}

// ---------------------------------------------------------------------------
// Narrative — the detail page's "what happened next".
// ---------------------------------------------------------------------------

export interface GuardrailNarrative {
  tone: GuardrailTone;
  /** First line is the headline. */
  lines: string[];
}

type NarrativeFields = ActionFields &
  Pick<GuardrailEvent, "attempt" | "blocking" | "on_error" | "metadata">;

const WOULD_HAVE: Record<string, string> = {
  masked: "masked",
  overridden: "overridden",
  blocked: "blocked",
  warned: "warned",
  reask: "re-asked",
};

const ACTION_VERB: Record<string, string> = {
  block: "blocked",
  warn: "warned",
  mask: "masked",
  override: "overridden",
  reask: "re-asked",
};

const PASSED_LINE = "The guardrail passed — content was allowed without modification.";
const WARNED_LINE =
  "The guardrail warned but did not block — content passed through unchanged with a warning logged.";
const REWROTE_LINE = "The guardrail rewrote the content before it continued.";

export function guardrailNarrative(ev: NarrativeFields): GuardrailNarrative {
  const effect = ev.effect;
  if (!effect) return legacyNarrative(ev);

  if ((ev.outcome ?? "").toLowerCase() === "errored" || effect === "unchecked") {
    if (effect === "unchecked") {
      return {
        tone: "warning",
        lines: [
          "Could not run — allowed through unchecked (on_error=allow). A degraded pass, not a verdict.",
        ],
      };
    }
    if (effect === "observed") {
      return {
        tone: "warning",
        lines: [
          "Could not run — observe-only: nothing was enforced, content passed (on_error applies only to blocking rules).",
        ],
      };
    }
    return { tone: "danger", lines: ["Could not run — blocked (fail closed)."] };
  }

  switch (effect) {
    case "none":
      return { tone: "success", lines: [PASSED_LINE] };
    case "warned":
      return { tone: "warning", lines: [WARNED_LINE] };
    case "rewritten":
      return { tone: "info", lines: [REWROTE_LINE] };
    case "reasked":
      return {
        tone: "warning",
        lines: [
          ev.attempt != null
            ? `Model re-asked (attempt ${ev.attempt}); see the next attempt's events.`
            : "Model re-asked; see the next attempt's events.",
        ],
      };
    case "observed": {
      const verb =
        WOULD_HAVE[ev.action_taken ?? ""] ?? ACTION_VERB[ev.action ?? ""] ?? null;
      return {
        tone: "neutral",
        lines: [
          verb
            ? `Observe-only — would have ${verb}; nothing applied.`
            : "Observe-only — nothing applied.",
        ],
      };
    }
    case "blocked": {
      const lines = ["Run stopped here."];
      if (ev.final_reason) {
        lines.push(`Configured to ${ev.action ?? "block"}; ${ev.final_reason}`);
      } else if (ev.action && ev.action !== "block") {
        lines.push(`Configured to ${ev.action}.`);
      }
      const framework = (ev.metadata ?? {}).framework;
      if (typeof framework === "string" && framework) {
        lines.push(`The ${framework} framework call raised GuardrailBlocked.`);
      }
      return { tone: "danger", lines };
    }
  }
  return legacyNarrative(ev);
}

/** Rows written before 1.89.0: the outcome alone, read as it always was. */
function legacyNarrative(ev: NarrativeFields): GuardrailNarrative {
  switch ((ev.outcome ?? "").toLowerCase()) {
    case "blocked":
      return {
        tone: "danger",
        lines: [
          "The guardrail blocked this content. The agent's chain halted at this step and " +
            "propagated the block to the caller (or fell back to the configured handler, if any).",
        ],
      };
    case "filtered":
      return { tone: "info", lines: [REWROTE_LINE] };
    case "warned":
      return { tone: "warning", lines: [WARNED_LINE] };
    case "reasked":
      return { tone: "warning", lines: ["Model re-asked; see the next attempt's events."] };
    case "errored": {
      const onError = ev.on_error ?? (ev.metadata ?? {}).on_error;
      const failedOpen = onError === "allow";
      return {
        tone: "warning",
        lines: [
          "The check could not run (e.g. a model-judged detector's LLM call failed). " +
            `Its outcome reflects the guardrail's on_error=${String(onError ?? "block")} ` +
            "policy, not a real verdict: " +
            (failedOpen
              ? "content was allowed through (fail open)."
              : "content was blocked (fail closed) if this rule was blocking."),
        ],
      };
    }
    default:
      return { tone: "success", lines: [PASSED_LINE] };
  }
}
