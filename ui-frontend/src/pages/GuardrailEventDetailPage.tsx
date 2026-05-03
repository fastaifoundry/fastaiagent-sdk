/**
 * Guardrail Event Detail page (/guardrail-events/:eventId).
 *
 * Three panels — *what triggered it*, *which rule matched*, *what happened
 * next* — plus a conversation-context section underneath. The "Mark as
 * false positive" button toggles a flag stored on the event row so devs
 * can curate signal vs. noise without ever hand-editing the DB.
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
  Flag,
  RefreshCw,
  Shield,
} from "lucide-react";
import { toast } from "sonner";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import {
  useGuardrailEvent,
  useMarkFalsePositive,
} from "@/hooks/use-guardrails";
import { ApiError } from "@/lib/api";
import { formatTimeAgo } from "@/lib/format";
import type { GuardrailContextSpan, GuardrailEvent } from "@/lib/types";
import { cn } from "@/lib/utils";

const OUTCOME_META: Record<
  string,
  { label: string; className: string; icon: React.ComponentType<{ className?: string }> }
> = {
  passed: {
    label: "✓ passed",
    className: "bg-fa-success/10 text-fa-success border-fa-success/40",
    icon: Shield,
  },
  blocked: {
    label: "🚫 blocked",
    className: "bg-destructive/10 text-destructive border-destructive/40",
    icon: CircleSlash,
  },
  warned: {
    label: "⚠ warned",
    className: "bg-fa-warning/10 text-fa-warning border-fa-warning/40",
    icon: AlertTriangle,
  },
  filtered: {
    label: "✎ filtered",
    className: "bg-fa-warning/10 text-fa-warning border-fa-warning/40",
    icon: AlertTriangle,
  },
};

function OutcomeBadge({ outcome }: { outcome: string | null }) {
  const meta =
    OUTCOME_META[(outcome ?? "").toLowerCase()] ?? {
      label: outcome ?? "—",
      className: "bg-muted text-muted-foreground",
      icon: Shield,
    };
  const Icon = meta.icon;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 font-mono text-xs uppercase",
        meta.className,
      )}
    >
      <Icon className="h-3 w-3" />
      {meta.label}
    </span>
  );
}

export function GuardrailEventDetailPage() {
  const { eventId } = useParams<{ eventId: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const detail = useGuardrailEvent(eventId);
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

  const { event, trigger, context } = detail.data;

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
              {formatTimeAgo(event.timestamp)}
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
          <CardContent className="space-y-2">
            <div className="text-xs text-muted-foreground">
              <span className="font-mono uppercase">{trigger.content_type}</span>
              {trigger.span_name && (
                <> · from span <span className="font-mono">{trigger.span_name}</span></>
              )}
            </div>
            <pre className="max-h-64 overflow-auto rounded-md border bg-muted/20 p-2 font-mono text-xs whitespace-pre-wrap">
              {trigger.text ?? "(no payload captured — set FASTAIAGENT_TRACE_PAYLOADS=1)"}
            </pre>
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

function RuleDetail({ event }: { event: GuardrailEvent }) {
  const md = event.metadata ?? {};
  const judgePrompt = (md as Record<string, unknown>).judge_prompt;
  const judgeResponse = (md as Record<string, unknown>).judge_response;
  const matched = (md as Record<string, unknown>).match;
  const piiTypes = (md as Record<string, unknown>).pii_types;

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
      {event.score != null && (
        <li>
          <span className="text-muted-foreground">score:</span>{" "}
          <span className="font-mono">{event.score.toFixed(2)}</span>
        </li>
      )}
      {Array.isArray(piiTypes) && piiTypes.length > 0 && (
        <li>
          <span className="text-muted-foreground">pii types:</span>{" "}
          <span className="font-mono">{(piiTypes as string[]).join(", ")}</span>
        </li>
      )}
      {typeof matched === "string" && matched && (
        <li>
          <span className="text-muted-foreground">matched:</span>{" "}
          <span className="font-mono text-fa-warning">{matched}</span>
        </li>
      )}
      {typeof judgePrompt === "string" && judgePrompt && (
        <li className="space-y-1">
          <span className="text-muted-foreground">judge prompt:</span>
          <pre className="max-h-32 overflow-auto rounded-md border bg-muted/20 p-2 font-mono text-xs whitespace-pre-wrap">
            {judgePrompt}
          </pre>
        </li>
      )}
      {typeof judgeResponse === "string" && judgeResponse && (
        <li className="space-y-1">
          <span className="text-muted-foreground">judge response:</span>
          <pre className="max-h-32 overflow-auto rounded-md border bg-muted/20 p-2 font-mono text-xs whitespace-pre-wrap">
            {judgeResponse}
          </pre>
        </li>
      )}
      {Object.keys(md).length === 0 && (
        <li className="text-xs text-muted-foreground">
          No additional metadata — the guardrail only emitted its
          name + outcome.
        </li>
      )}
    </ul>
  );
}

function OutcomeDetail({ event }: { event: GuardrailEvent }) {
  const md = event.metadata ?? {};
  const before = (md as Record<string, unknown>).before;
  const after = (md as Record<string, unknown>).after;

  if (event.outcome === "blocked") {
    return (
      <div className="space-y-2 text-sm">
        <p>
          The guardrail <span className="font-mono">blocked</span> this content.
          The agent's chain halted at this step and propagated the block to the
          caller (or fell back to the configured handler, if any).
        </p>
        {event.message && (
          <pre className="rounded-md border border-destructive/40 bg-destructive/5 p-2 font-mono text-xs whitespace-pre-wrap text-destructive">
            {event.message}
          </pre>
        )}
      </div>
    );
  }

  if (
    event.outcome === "filtered" ||
    (typeof before === "string" && typeof after === "string")
  ) {
    return (
      <div className="space-y-2">
        <p className="text-sm">
          The guardrail <span className="font-mono">rewrote</span> the content
          before it continued.
        </p>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <div>
            <div className="mb-1 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
              before
            </div>
            <pre className="max-h-40 overflow-auto rounded-md border border-destructive/30 bg-destructive/5 p-2 font-mono text-xs whitespace-pre-wrap">
              {String(before ?? "(unavailable)")}
            </pre>
          </div>
          <div>
            <div className="mb-1 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
              after
            </div>
            <pre className="max-h-40 overflow-auto rounded-md border border-fa-success/30 bg-fa-success/5 p-2 font-mono text-xs whitespace-pre-wrap">
              {String(after ?? "(unavailable)")}
            </pre>
          </div>
        </div>
      </div>
    );
  }

  if (event.outcome === "warned") {
    return (
      <p className="text-sm">
        The guardrail <span className="font-mono">warned</span> but did not
        block — content passed through unchanged with a warning logged.
      </p>
    );
  }

  return (
    <p className="text-sm">
      The guardrail <span className="font-mono">passed</span> — content was
      allowed without modification.
    </p>
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
                {formatTimeAgo(s.start_time)}
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
