/**
 * Guardrail Event Detail page (/guardrail-events/:eventId).
 *
 * Three panels — *what triggered it*, *which rule matched*, *what happened
 * next* — then the evidence the check recorded and a conversation-context
 * section underneath. The "Mark as false positive" button toggles a flag
 * stored on the event row so devs can curate signal vs. noise without ever
 * hand-editing the DB.
 *
 * 1.89.0 — Panel 1 shows exactly the text the rule judged (``judged``), and
 * the narrative reads what the check did to the run (``effect``) together
 * with ``blocking`` / ``on_error`` / ``final_reason``, not the verdict alone.
 *
 * Sprint 2 / Feature 3.
 */
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  ChevronLeft,
  CircleSlash,
  ExternalLink,
  EyeOff,
  Flag,
  PenLine,
  RefreshCw,
  Shield,
} from "lucide-react";
import { toast } from "sonner";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { GuardrailEvidence } from "@/components/guardrails/GuardrailEvidence";
import {
  useGuardrailEvent,
  useMarkFalsePositive,
} from "@/hooks/use-guardrails";
import { ApiError } from "@/lib/api";
import {
  actionDiverged,
  EFFECT_META,
  guardrailNarrative,
  outcomeMeta,
  TONE_BORDER,
  TONE_CHIP,
  type GuardrailTone,
} from "@/lib/guardrail";
import type {
  GuardrailContextSpan,
  GuardrailEvent,
  GuardrailEventDetail,
} from "@/lib/types";
import { cn } from "@/lib/utils";
import { Timestamp } from "@/components/shared/Timestamp";

const TONE_ICON: Record<GuardrailTone, React.ComponentType<{ className?: string }>> = {
  success: Shield,
  warning: AlertTriangle,
  danger: CircleSlash,
  info: PenLine,
  neutral: Shield,
};

function ToneBadge({ label, tone }: { label: string; tone: GuardrailTone }) {
  const Icon = TONE_ICON[tone];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 font-mono text-xs uppercase",
        TONE_CHIP[tone],
        TONE_BORDER[tone],
      )}
    >
      <Icon className="h-3 w-3" />
      {label}
    </span>
  );
}

function OutcomeBadge({ outcome }: { outcome: string | null }) {
  const meta = outcomeMeta(outcome);
  return <ToneBadge label={meta.label} tone={meta.tone} />;
}

export function GuardrailEventDetailPage() {
  const { eventId } = useParams<{ eventId: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  // Read-mode redaction of the judged text and the before/after diff; the
  // backend honors it only when a read-mode RedactionPolicy is installed.
  const [redact, setRedact] = useState(false);
  const detail = useGuardrailEvent(eventId, redact);
  const mark = useMarkFalsePositive();
  const [busy, setBusy] = useState(false);

  if (detail.isLoading) return <TableSkeleton rows={6} />;
  if (detail.error || !detail.data) {
    return (
      <EmptyState
        title="Event not found"
        description="The event id may belong to another project, or the event was never recorded."
        icon={Shield}
      />
    );
  }

  const { event, context } = detail.data;
  const effectMeta = event.effect ? EFFECT_META[event.effect] : null;

  const handleToggleFalsePositive = async () => {
    if (!eventId) return;
    setBusy(true);
    try {
      const next = !event.false_positive;
      await mark.mutateAsync({ eventId, falsePositive: next });
      toast.success(
        next
          ? "Marked as false positive — flag persists across refresh"
          : "Cleared false-positive flag",
      );
      queryClient.invalidateQueries({ queryKey: ["guardrail-event", eventId] });
      queryClient.invalidateQueries({ queryKey: ["guardrails"] });
    } catch (e) {
      if (e instanceof ApiError) toast.error(e.message);
      else toast.error("Update failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-5">
      <PageHeader
        title={event.guardrail_name}
        description={`${event.guardrail_type ?? "guardrail"} · ${event.position ?? "?"}`}
      >
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigate("/guardrails")}
        >
          <ChevronLeft className="mr-1.5 h-3.5 w-3.5" />
          Back
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => detail.refetch()}
          disabled={detail.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${detail.isFetching ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
        <label
          className="ml-1 flex items-center gap-2 rounded-md border px-2.5 py-1 text-xs font-mono"
          title={
            "Sends ?redact=true to the guardrail event API. Honored only when a " +
            "RedactionPolicy(mode in {read, both}) is installed via " +
            "fastaiagent.trace.set_redaction_policy(...). See docs/security.md."
          }
        >
          <EyeOff className="h-3.5 w-3.5 text-muted-foreground" />
          <span>Mask secrets</span>
          <Switch
            checked={redact}
            onCheckedChange={setRedact}
            aria-label="Toggle guardrail event redaction"
            size="sm"
          />
        </label>
        <Button
          variant={event.false_positive ? "default" : "outline"}
          size="sm"
          onClick={handleToggleFalsePositive}
          disabled={busy || mark.isPending}
          data-testid="mark-false-positive-button"
        >
          <Flag className="mr-1.5 h-3.5 w-3.5" />
          {event.false_positive ? "Marked false positive" : "Mark as false positive"}
        </Button>
      </PageHeader>

      {/* ─── Header summary ────────────────────────────────────────── */}
      <Card>
        <CardContent className="space-y-2 pt-6">
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <OutcomeBadge outcome={event.outcome} />
            {effectMeta && (
              <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                effect
                <ToneBadge label={effectMeta.label} tone={effectMeta.tone} />
              </span>
            )}
            {event.false_positive && (
              <span className="inline-flex items-center gap-1.5 rounded-md border border-primary/40 bg-primary/10 px-2 py-0.5 font-mono text-xs uppercase text-primary">
                <Flag className="h-3 w-3" />
                false positive
              </span>
            )}
            <span className="font-mono text-xs text-muted-foreground">
              event id: {event.event_id}
            </span>
            <span className="text-xs text-muted-foreground">
              <Timestamp iso={event.timestamp} />
            </span>
            {event.agent_name && (
              <Link
                to={`/agents/${encodeURIComponent(event.agent_name)}`}
                className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
              >
                <ExternalLink className="h-3 w-3" />
                {event.agent_name}
              </Link>
            )}
            {event.trace_id && (
              <Link
                to={`/traces/${event.trace_id}`}
                className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
              >
                <ExternalLink className="h-3 w-3" />
                trace
              </Link>
            )}
          </div>
          {event.message && (
            <p className="text-sm text-muted-foreground">{event.message}</p>
          )}
        </CardContent>
      </Card>

      {/* ─── Three panels ──────────────────────────────────────────── */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card data-testid="panel-trigger">
          <CardHeader>
            <CardTitle className="text-sm">// 1 — WHAT TRIGGERED IT</CardTitle>
          </CardHeader>
          <CardContent>
            <TriggerDetail detail={detail.data} />
          </CardContent>
        </Card>

        <Card data-testid="panel-rule">
          <CardHeader>
            <CardTitle className="text-sm">// 2 — WHICH RULE MATCHED</CardTitle>
          </CardHeader>
          <CardContent>
            <RuleDetail event={event} />
          </CardContent>
        </Card>

        <Card data-testid="panel-outcome">
          <CardHeader>
            <CardTitle className="text-sm">// 3 — WHAT HAPPENED NEXT</CardTitle>
          </CardHeader>
          <CardContent>
            <OutcomeDetail event={event} />
          </CardContent>
        </Card>
      </div>

      {/* ─── Evidence ──────────────────────────────────────────────── */}
      <Card data-testid="panel-evidence">
        <CardHeader>
          <CardTitle className="text-sm">// EVIDENCE</CardTitle>
        </CardHeader>
        <CardContent>
          <GuardrailEvidence event={event} />
        </CardContent>
      </Card>

      {/* ─── Context section ───────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">// EXECUTION CONTEXT</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {context.spans.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              No spans recorded for this trace — turn on tracing
              (default) and re-run the agent.
            </p>
          ) : (
            <ContextTimeline spans={context.spans} highlightSpanId={event.span_id} />
          )}

          <div>
            <div className="mb-2 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
              // OTHER GUARDRAILS ON THE SAME CONTENT
            </div>
            {context.sibling_events.length === 0 ? (
              <p className="text-xs text-muted-foreground">
                No other guardrails ran on this content.
              </p>
            ) : (
              <ul className="space-y-1.5">
                {context.sibling_events.map((s) => (
                  <li key={s.event_id} className="flex items-center gap-2 text-xs">
                    <Link
                      to={`/guardrail-events/${s.event_id}`}
                      className="font-mono hover:text-primary"
                    >
                      {s.guardrail_name}
                    </Link>
                    <OutcomeBadge outcome={s.outcome} />
                    {s.message && (
                      <span className="text-muted-foreground">— {s.message}</span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

/**
 * Panel 1. ``judged`` is exactly what the rule saw (1.89.0). Older rows fall
 * back to the parent span's payload, which may hold the text after a rewrite;
 * with neither, say so rather than guess.
 */
function TriggerDetail({ detail }: { detail: GuardrailEventDetail }) {
  const { event, trigger, context } = detail;
  const judged = detail.judged ?? { text: null, recorded: false };

  // An earlier rule in the same pass rewrote the payload this rule judged.
  const rewrittenBy = event.batch_id
    ? context.sibling_events.find(
        (s) =>
          s.batch_id === event.batch_id &&
          (s.outcome ?? "").toLowerCase() === "filtered" &&
          !!s.timestamp &&
          !!event.timestamp &&
          s.timestamp < event.timestamp,
      )
    : undefined;

  let source: "judged" | "span" | null = null;
  let text: string | null = null;
  if (judged.recorded) {
    source = "judged";
    text = judged.text ?? "";
  } else if (trigger.text != null) {
    source = "span";
    text = trigger.text;
  }

  return (
    <div className="space-y-2">
      <div className="text-xs text-muted-foreground">
        <span className="font-mono uppercase">{trigger.content_type}</span>
        {source === "judged" && <> · exactly the text the rule judged</>}
        {source === "span" && trigger.span_name && (
          <> · from span <span className="font-mono">{trigger.span_name}</span></>
        )}
      </div>
      {rewrittenBy && (
        <p className="text-xs text-fa-info" data-testid="judged-after-rewrite">
          Judged after{" "}
          <Link
            to={`/guardrail-events/${rewrittenBy.event_id}`}
            className="font-mono hover:underline"
          >
            {rewrittenBy.guardrail_name}
          </Link>{" "}
          rewrote it.
        </p>
      )}
      {source == null ? (
        <p className="text-xs text-muted-foreground" data-testid="judged-not-recorded">
          Not recorded — this event predates 1.89.0.
        </p>
      ) : (
        <pre className="max-h-64 overflow-auto rounded-md border bg-muted/20 p-2 font-mono text-xs whitespace-pre-wrap">
          {text || "(empty)"}
        </pre>
      )}
    </div>
  );
}

function RuleDetail({ event }: { event: GuardrailEvent }) {
  const diverged = actionDiverged(event);
  const chain = [
    event.action,
    event.action_taken,
    event.effect ? EFFECT_META[event.effect].label : null,
  ];
  const showChain = chain.some((c) => c != null);
  const blocking = event.blocking;

  return (
    <ul className="space-y-1.5 text-sm">
      <li>
        <span className="text-muted-foreground">name:</span>{" "}
        <span className="font-mono">{event.guardrail_name}</span>
      </li>
      <li>
        <span className="text-muted-foreground">type:</span>{" "}
        <span className="font-mono">{event.guardrail_type ?? "—"}</span>
      </li>
      <li>
        <span className="text-muted-foreground">position:</span>{" "}
        <span className="font-mono">{event.position ?? "—"}</span>
      </li>
      {showChain && (
        <li data-testid="rule-action-chain">
          <span className="text-muted-foreground">action → taken → effect:</span>{" "}
          <span
            className={cn("font-mono", diverged && "text-fa-warning")}
            title={
              diverged
                ? (event.final_reason ??
                  "The configured action and what the check did differ")
                : undefined
            }
          >
            {chain.map((c) => c ?? "—").join(" → ")}
            {diverged && " *"}
          </span>
        </li>
      )}
      {event.final_reason && (
        <li>
          <span className="text-muted-foreground">final reason:</span>{" "}
          <span className="font-mono">{event.final_reason}</span>
        </li>
      )}
      <li>
        <span className="text-muted-foreground">severity:</span>{" "}
        <span className="font-mono">{event.severity ?? "—"}</span>
        {event.floor && (
          <span
            className="ml-1.5 rounded-sm bg-primary/10 px-1 py-px text-[9px] uppercase text-primary"
            title="Organisation baseline — only a domain admin may change this rule"
          >
            floor
          </span>
        )}
      </li>
      {blocking != null && (
        <li>
          <span className="text-muted-foreground">mode:</span>{" "}
          <span className="font-mono">{blocking ? "blocking" : "observe-only"}</span>
        </li>
      )}
      {event.on_error && (
        <li>
          <span className="text-muted-foreground">on_error:</span>{" "}
          <span className="font-mono">{event.on_error}</span>
        </li>
      )}
      {event.origin && (
        <li>
          <span className="text-muted-foreground">origin:</span>{" "}
          <span className="font-mono">{event.origin}</span>
        </li>
      )}
      {event.latency_ms != null && (
        <li>
          <span className="text-muted-foreground">latency:</span>{" "}
          <span className="font-mono">{event.latency_ms} ms</span>
        </li>
      )}
      {event.attempt != null && (
        <li>
          <span className="text-muted-foreground">attempt:</span>{" "}
          <span className="font-mono">{event.attempt}</span>
        </li>
      )}
      {event.score != null && (
        <li>
          <span className="text-muted-foreground">score:</span>{" "}
          <span className="font-mono">{event.score.toFixed(2)}</span>
        </li>
      )}
      {event.rule?.description && (
        <li className="text-xs text-muted-foreground">{event.rule.description}</li>
      )}
      {event.effect == null && (
        <li className="text-xs text-muted-foreground">
          Recorded before 1.89.0 — the rule's mode, on_error and effect were not stored.
        </li>
      )}
    </ul>
  );
}

function DiffColumn({
  label,
  value,
  className,
}: {
  label: string;
  value: unknown;
  className: string;
}) {
  return (
    <div>
      <div className="mb-1 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        {label}
      </div>
      <pre
        className={cn(
          "max-h-40 overflow-auto rounded-md border p-2 font-mono text-xs whitespace-pre-wrap",
          className,
        )}
      >
        {String(value ?? "(unavailable)")}
      </pre>
    </div>
  );
}

function OutcomeDetail({ event }: { event: GuardrailEvent }) {
  const md = event.metadata ?? {};
  const before = (md as Record<string, unknown>).before;
  const after = (md as Record<string, unknown>).after;
  const narrative = guardrailNarrative(event);
  const outcome = (event.outcome ?? "").toLowerCase();

  // A rewrite that was applied shows before/after; one that was not (an
  // observe-only rule, a rewrite the caller refused) shows what it would
  // have produced.
  const hasDiff = typeof before === "string" && typeof after === "string";
  const applied =
    event.effect === "rewritten" || (event.effect == null && outcome === "filtered");
  const showDiff = hasDiff || applied;

  const messageClass =
    narrative.tone === "danger"
      ? "border-destructive/40 bg-destructive/5 text-destructive"
      : "border-fa-warning/40 bg-fa-warning/5";
  const showMessage =
    !!event.message &&
    (outcome === "blocked" || outcome === "errored" || event.effect === "blocked");

  return (
    <div className="space-y-2 text-sm" data-testid="outcome-narrative">
      {narrative.lines.map((line, i) => (
        <p key={i} className={cn(i > 0 && "text-muted-foreground")}>
          {line}
        </p>
      ))}
      {showMessage && (
        <pre
          className={cn(
            "rounded-md border p-2 font-mono text-xs whitespace-pre-wrap",
            messageClass,
          )}
        >
          {event.message}
        </pre>
      )}
      {showDiff && (
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <DiffColumn
            label="before"
            value={before}
            className="border-destructive/30 bg-destructive/5"
          />
          <DiffColumn
            label={applied ? "after" : "would have produced (not applied)"}
            value={after}
            className={
              applied ? "border-fa-success/30 bg-fa-success/5" : "border-dashed bg-muted/20"
            }
          />
        </div>
      )}
    </div>
  );
}

function ContextTimeline({
  spans,
  highlightSpanId,
}: {
  spans: GuardrailContextSpan[];
  highlightSpanId: string | null;
}) {
  return (
    <ol className="relative space-y-3 border-l-2 border-border pl-4">
      {spans.map((s) => {
        const active = highlightSpanId === s.span_id;
        return (
          <li key={s.span_id} className="relative">
            <span
              className={cn(
                "absolute -left-[21px] top-1 h-3 w-3 rounded-full border-2",
                active
                  ? "border-primary bg-primary"
                  : "border-muted-foreground bg-card",
              )}
            />
            <div className="flex items-center justify-between gap-2 text-xs">
              <span className="font-mono">{s.name}</span>
              <span className="text-muted-foreground">
                <Timestamp iso={s.start_time} />
              </span>
            </div>
            {(s.input || s.output) && (
              <div className="mt-1 grid grid-cols-1 gap-2 sm:grid-cols-2">
                {s.input && (
                  <pre className="max-h-24 overflow-auto rounded-md border bg-muted/20 p-2 font-mono text-[11px] whitespace-pre-wrap">
                    {s.input}
                  </pre>
                )}
                {s.output && (
                  <pre className="max-h-24 overflow-auto rounded-md border bg-muted/20 p-2 font-mono text-[11px] whitespace-pre-wrap">
                    {s.output}
                  </pre>
                )}
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}
