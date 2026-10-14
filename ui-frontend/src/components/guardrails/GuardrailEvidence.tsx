/**
 * What a check found, one renderer per guardrail type.
 *
 * Each renderer reads only keys a runner actually writes into the event's
 * ``metadata`` (or the rule snapshot's ``config``), so nothing here promises
 * a field that never arrives. Whatever is recorded is always reachable under
 * "All recorded metadata".
 */
import { Fragment, type ReactNode } from "react";
import { JsonViewer } from "@/components/shared/JsonViewer";
import type { GuardrailEvent } from "@/lib/types";
import { cn } from "@/lib/utils";

type Meta = Record<string, unknown>;

function strings(v: unknown): string[] | null {
  return Array.isArray(v) ? v.map((x) => String(x)) : null;
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function fmt(v: number | null, digits = 2): string {
  return v == null ? "—" : v.toFixed(digits);
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <li>
      <span className="text-muted-foreground">{label}:</span>{" "}
      <span className="font-mono">{children}</span>
    </li>
  );
}

function List({
  label,
  values,
  highlight = false,
  emptyText = "none",
}: {
  label: string;
  values: string[] | null;
  highlight?: boolean;
  emptyText?: string;
}) {
  if (values == null) return null;
  return (
    <li>
      <span className="text-muted-foreground">{label}:</span>{" "}
      {values.length > 0 ? (
        <span className={cn("font-mono", highlight && "text-fa-warning")}>
          {values.join(", ")}
        </span>
      ) : (
        <span className="font-mono text-muted-foreground">{emptyText}</span>
      )}
    </li>
  );
}

function Pre({ children }: { children: ReactNode }) {
  return (
    <pre className="max-h-48 overflow-auto rounded-md border bg-muted/20 p-2 font-mono text-xs whitespace-pre-wrap">
      {children}
    </pre>
  );
}

// ---------------------------------------------------------------------------
// content_safety — per-category score against its threshold.
// ---------------------------------------------------------------------------

function ContentSafetyEvidence({ md }: { md: Meta }) {
  const scores = (md.scores ?? {}) as Record<string, number>;
  const thresholds = (md.thresholds ?? {}) as Record<string, number>;
  const tripped = new Set(strings(md.tripped) ?? []);
  const unscored = strings(md.unscored) ?? [];
  const categories = Array.from(
    new Set([...Object.keys(scores), ...Object.keys(thresholds), ...tripped]),
  ).sort();
  return (
    <div className="space-y-2 text-sm" data-testid="evidence-content-safety">
      {categories.length > 0 && (
        <table className="w-full text-xs">
          <thead className="text-left text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
            <tr>
              <th className="py-1 font-normal">Category</th>
              <th className="py-1 text-right font-normal">Score</th>
              <th className="py-1 text-right font-normal">Threshold</th>
              <th className="py-1 text-right font-normal">Tripped</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {categories.map((c) => (
              <tr key={c} className={cn(tripped.has(c) && "text-destructive")}>
                <td className="py-0.5">{c}</td>
                <td className="py-0.5 text-right tabular-nums">{fmt(num(scores[c]))}</td>
                <td className="py-0.5 text-right tabular-nums">
                  {num(thresholds[c]) != null ? fmt(num(thresholds[c])) : "default"}
                </td>
                <td className="py-0.5 text-right">{tripped.has(c) ? "yes" : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {unscored.length > 0 && (
        <p className="text-xs text-fa-warning">
          Not scored by the judge: <span className="font-mono">{unscored.join(", ")}</span>
        </p>
      )}
      {typeof md.backend === "string" && (
        <ul className="space-y-1 text-xs">
          <Row label="backend">{md.backend}</Row>
        </ul>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// groundedness — score vs threshold, and the claims the context did not support.
// ---------------------------------------------------------------------------

function GroundednessEvidence({ md }: { md: Meta }) {
  const claims = strings(md.unsupported_claims);
  return (
    <div className="space-y-2 text-sm" data-testid="evidence-groundedness">
      <ul className="space-y-1.5">
        <Row label="score">{fmt(num(md.score))}</Row>
        <Row label="threshold">{fmt(num(md.threshold))}</Row>
      </ul>
      {claims && (
        <div>
          <div className="mb-1 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
            unsupported claims
          </div>
          {claims.length === 0 ? (
            <p className="text-xs text-muted-foreground">None.</p>
          ) : (
            <ul className="list-disc space-y-1 pl-5 text-xs">
              {claims.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// schema — JSON-Schema violations.
// ---------------------------------------------------------------------------

interface Violation {
  path?: string;
  message?: string;
  validator?: string;
}

function SchemaEvidence({ md }: { md: Meta }) {
  const violations = Array.isArray(md.violations) ? (md.violations as Violation[]) : null;
  if (!violations) return null;
  return (
    <table className="w-full text-xs" data-testid="evidence-schema">
      <thead className="text-left text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        <tr>
          <th className="py-1 font-normal">Path</th>
          <th className="py-1 font-normal">Message</th>
          <th className="py-1 font-normal">Validator</th>
        </tr>
      </thead>
      <tbody>
        {violations.map((v, i) => (
          <tr key={i} className="align-top">
            <td className="py-0.5 pr-2 font-mono">{v.path || "(root)"}</td>
            <td className="py-0.5 pr-2">{v.message ?? "—"}</td>
            <td className="py-0.5 font-mono">{v.validator ?? "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------------------
// classifier — keyword categories detected, and those that block.
// ---------------------------------------------------------------------------

function ClassifierEvidence({ md }: { md: Meta }) {
  return (
    <ul className="space-y-1.5 text-sm" data-testid="evidence-classifier">
      <List label="detected" values={strings(md.detected)} />
      <List label="blocked" values={strings(md.blocked)} highlight />
    </ul>
  );
}

// ---------------------------------------------------------------------------
// llm_judge — chat (prompt + reply) or the Decisions backend (probability).
// ---------------------------------------------------------------------------

function parseJudgeReply(reply: string): { verdict?: string; reason?: string } | null {
  const match = reply.match(/\{[\s\S]*?\}/);
  if (!match) return null;
  try {
    const parsed = JSON.parse(match[0]) as Record<string, unknown>;
    if (typeof parsed !== "object" || parsed == null) return null;
    return {
      verdict: parsed.verdict != null ? String(parsed.verdict) : undefined,
      reason: parsed.reason != null ? String(parsed.reason) : undefined,
    };
  } catch {
    return null;
  }
}

function LlmJudgeEvidence({ event, md }: { event: GuardrailEvent; md: Meta }) {
  const config = event.rule?.config ?? {};
  if (md.backend === "decisions" || config.backend === "decisions") {
    return (
      <ul className="space-y-1.5 text-sm" data-testid="evidence-llm-judge-decisions">
        <Row label="backend">decisions</Row>
        <Row label="probability">{fmt(num(md.probability), 3)}</Row>
        <Row label="threshold">{fmt(num(md.threshold), 3)}</Row>
        {typeof config.instructions === "string" && (
          <li className="space-y-1">
            <span className="text-muted-foreground">instructions:</span>
            <Pre>{config.instructions}</Pre>
          </li>
        )}
      </ul>
    );
  }
  const prompt = typeof config.prompt === "string" ? config.prompt : null;
  const reply = event.message;
  const parsed = reply ? parseJudgeReply(reply) : null;
  return (
    <div className="space-y-2 text-sm" data-testid="evidence-llm-judge">
      <div className="space-y-1">
        <span className="text-muted-foreground">judge prompt:</span>
        {prompt ? (
          <Pre>{prompt}</Pre>
        ) : (
          <p className="text-xs text-muted-foreground">
            The rule's built-in default (no prompt configured).
          </p>
        )}
      </div>
      {reply && (
        <div className="space-y-1">
          <span className="text-muted-foreground">judge reply:</span>
          <Pre>{reply}</Pre>
        </div>
      )}
      {parsed && (
        <ul className="space-y-1.5">
          {parsed.verdict && <Row label="verdict">{parsed.verdict}</Row>}
          {parsed.reason && <Row label="reason">{parsed.reason}</Row>}
        </ul>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// regex — the pattern from the rule.
// ---------------------------------------------------------------------------

function RegexEvidence({ event }: { event: GuardrailEvent }) {
  const config = event.rule?.config ?? {};
  if (typeof config.pattern !== "string") return null;
  return (
    <ul className="space-y-1.5 text-sm" data-testid="evidence-regex">
      <Row label="pattern">{config.pattern}</Row>
      <Row label="should match">{String(Boolean(config.should_match))}</Row>
      {config.case_insensitive != null && (
        <Row label="case insensitive">{String(Boolean(config.case_insensitive))}</Row>
      )}
    </ul>
  );
}

// ---------------------------------------------------------------------------
// topic — the operator's named topics, and the polarity that made a verdict.
// ---------------------------------------------------------------------------

function TopicEvidence({ md }: { md: Meta }) {
  const mode = md.mode;
  const topicsAsked = strings(md.topics);
  return (
    <ul className="space-y-1.5 text-sm" data-testid="evidence-topic">
      {typeof mode === "string" && (
        <Row label="mode">{mode === "deny" ? "deny (blocklist)" : "allow (on-topic gate)"}</Row>
      )}
      <List label="topics matched" values={strings(md.matched)} highlight />
      {topicsAsked && topicsAsked.length > 0 && (
        <Row label="topics checked">{topicsAsked.join(", ")}</Row>
      )}
      {typeof md.backend === "string" && <Row label="backend">{md.backend}</Row>}
    </ul>
  );
}

// ---------------------------------------------------------------------------
// pii / secrets — entity kinds and counts, never the matched values.
// ---------------------------------------------------------------------------

function Counts({ counts }: { counts: unknown }) {
  if (counts == null || typeof counts !== "object") return null;
  return (
    <Row label="counts">
      {Object.entries(counts as Record<string, number>)
        .map(([k, n]) => `${k} ×${n}`)
        .join(", ") || "—"}
    </Row>
  );
}

function EntityEvidence({ event, md }: { event: GuardrailEvent; md: Meta }) {
  return (
    <ul className="space-y-1.5 text-sm" data-testid="evidence-entities">
      <List label="found" values={strings(md.found)} highlight />
      <Counts counts={md.counts} />
      {event.guardrail_type === "pii" && typeof md.backend === "string" && (
        <Row label="backend">{md.backend}</Row>
      )}
      {event.guardrail_type === "pii" && strings(md.entities) && (
        <Row label="entities checked">{strings(md.entities)!.join(", ") || "—"}</Row>
      )}
    </ul>
  );
}

// ---------------------------------------------------------------------------
// code — the built-in factories (no_pii, no_secrets, toxicity_check, …) write
// these keys; a custom ``fn=`` writes whatever it likes (raw view below).
// ---------------------------------------------------------------------------

function builtinEvidence(md: Meta): ReactNode {
  const items: ReactNode[] = [];
  const add = (key: string, node: ReactNode) => items.push(<Fragment key={key}>{node}</Fragment>);
  if (strings(md.pii_types)) add("pii", <List label="pii types" values={strings(md.pii_types)} highlight />);
  if (md.counts != null) add("counts", <Counts counts={md.counts} />);
  if (strings(md.secret_kinds))
    add("secrets", <List label="secret kinds" values={strings(md.secret_kinds)} highlight />);
  if (strings(md.matched_patterns))
    add("patterns", <List label="matched patterns" values={strings(md.matched_patterns)} highlight />);
  if (strings(md.flagged_categories))
    add("flagged", <List label="flagged categories" values={strings(md.flagged_categories)} highlight />);
  if (num(md.toxicity_score) != null)
    add("tox", <Row label="toxicity score">{fmt(num(md.toxicity_score))}</Row>);
  if (strings(md.toxic_words)) add("words", <List label="toxic words" values={strings(md.toxic_words)} highlight />);
  if (strings(md.blocked_domains))
    add("domains", <List label="blocked domains" values={strings(md.blocked_domains)} highlight />);
  if (num(md.spent_usd) != null || num(md.max_usd) != null)
    add(
      "cost",
      <Row label="spent / max">
        ${fmt(num(md.spent_usd), 4)} / ${fmt(num(md.max_usd), 4)}
      </Row>,
    );
  if (strings(md.banned_topics)) add("banned", <List label="banned topics" values={strings(md.banned_topics)} highlight />);
  if (strings(md.allowed_topics)) add("allowed", <List label="allowed topics" values={strings(md.allowed_topics)} />);
  if (strings(md.matched_topics)) add("matched", <List label="matched topics" values={strings(md.matched_topics)} />);
  if (items.length === 0) return null;
  return (
    <ul className="space-y-1.5 text-sm" data-testid="evidence-builtin">
      {items}
    </ul>
  );
}

const has = (md: Meta, ...keys: string[]) => keys.some((k) => md[k] != null);

/** The type's renderer, or null when the row holds none of what it reads. */
function renderTyped(event: GuardrailEvent, md: Meta): ReactNode {
  switch (event.guardrail_type) {
    case "content_safety":
      return has(md, "scores", "thresholds", "tripped", "unscored") ? (
        <ContentSafetyEvidence md={md} />
      ) : null;
    case "groundedness":
      return has(md, "score", "threshold", "unsupported_claims") ? (
        <GroundednessEvidence md={md} />
      ) : null;
    case "schema":
      return Array.isArray(md.violations) ? <SchemaEvidence md={md} /> : null;
    case "classifier":
      return has(md, "detected", "blocked") ? <ClassifierEvidence md={md} /> : null;
    case "llm_judge":
      return <LlmJudgeEvidence event={event} md={md} />;
    case "regex":
      return typeof event.rule?.config?.pattern === "string" ? (
        <RegexEvidence event={event} />
      ) : null;
    case "topic":
      return has(md, "mode", "matched", "topics") ? <TopicEvidence md={md} /> : null;
    case "pii":
    case "secrets":
      return has(md, "found", "counts") ? <EntityEvidence event={event} md={md} /> : null;
    default:
      // ``grounded()`` is a code rule with the groundedness shape.
      if (md.unsupported_claims != null) return <GroundednessEvidence md={md} />;
      return builtinEvidence(md);
  }
}

export function GuardrailEvidence({ event }: { event: GuardrailEvent }) {
  const md = (event.metadata ?? {}) as Meta;
  const typed = renderTyped(event, md);
  const error = typeof md.error === "string" ? md.error : null;
  const actionError = typeof md.action_error === "string" ? md.action_error : null;
  const hasMetadata = Object.keys(md).length > 0;

  return (
    <div className="space-y-3">
      {(error || actionError) && (
        <ul className="space-y-1.5 text-sm">
          {error && (
            <li>
              <span className="text-muted-foreground">error:</span>{" "}
              <span className="font-mono text-fa-warning">{error}</span>
            </li>
          )}
          {actionError && (
            <li>
              <span className="text-muted-foreground">action error:</span>{" "}
              <span className="font-mono text-fa-warning">{actionError}</span>
            </li>
          )}
        </ul>
      )}
      {typed ??
        (!error && !actionError && (
          <p className="text-xs text-muted-foreground">
            {hasMetadata
              ? "No type-specific evidence for this rule — see the recorded metadata below."
              : "No additional metadata — the guardrail only emitted its name and outcome."}
          </p>
        ))}
      {hasMetadata && (
        <details data-testid="evidence-raw-metadata">
          <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground">
            All recorded metadata
          </summary>
          <JsonViewer data={md} className="mt-2" />
        </details>
      )}
    </div>
  );
}
