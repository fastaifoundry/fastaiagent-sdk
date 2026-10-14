import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type {
  FalsePositiveResponse,
  GuardrailEventDetail,
  GuardrailEventsPage,
} from "@/lib/types";

/** Rows per page on the events list. */
export const GUARDRAIL_EVENTS_PAGE_SIZE = 50;

interface Filters {
  rule?: string | null;
  outcome?: string | null;
  agent?: string | null;
  type?: string | null;
  position?: string | null;
  false_positive?: boolean | null;
  since?: string;
  until?: string;
  /** Leave out clean passes (1.89.0). */
  hide_passes?: boolean;
  /** Leave out replay, Playground and other non-production checks (1.89.0). */
  exclude_experiments?: boolean;
  page?: number;
  page_size?: number;
}

function buildQuery(f: Filters): string {
  const params = new URLSearchParams();
  if (f.rule) params.set("rule", f.rule);
  if (f.outcome) params.set("outcome", f.outcome);
  if (f.agent) params.set("agent", f.agent);
  if (f.type) params.set("type", f.type);
  if (f.position) params.set("position", f.position);
  if (f.false_positive != null) {
    params.set("false_positive", String(f.false_positive));
  }
  if (f.since) params.set("since", f.since);
  if (f.until) params.set("until", f.until);
  if (f.hide_passes) params.set("hide_passes", "true");
  if (f.exclude_experiments) params.set("exclude_experiments", "true");
  params.set("page", String(f.page ?? 1));
  params.set("page_size", String(f.page_size ?? GUARDRAIL_EVENTS_PAGE_SIZE));
  return params.toString() ? `?${params.toString()}` : "";
}

export function useGuardrailEvents(filters: Filters = {}) {
  return useQuery({
    queryKey: ["guardrails", filters],
    queryFn: () =>
      api.get<GuardrailEventsPage>(`/guardrail-events${buildQuery(filters)}`),
  });
}

export function useGuardrailEvent(eventId: string | undefined, redact: boolean = false) {
  // ``?redact=true`` applies read-mode redaction to the judged text and the
  // before/after diff, as the trace API does (honored only when a
  // ``RedactionPolicy(mode in {"read", "both"})`` is installed).
  const suffix = redact ? "?redact=true" : "";
  return useQuery({
    queryKey: ["guardrail-event", eventId, redact],
    queryFn: () =>
      api.get<GuardrailEventDetail>(
        `/guardrail-events/${encodeURIComponent(eventId!)}${suffix}`,
      ),
    enabled: !!eventId,
  });
}

export function useMarkFalsePositive() {
  return useMutation({
    mutationFn: ({
      eventId,
      falsePositive,
      note,
    }: {
      eventId: string;
      falsePositive: boolean;
      note?: string;
    }) =>
      api.patch<FalsePositiveResponse>(
        `/guardrail-events/${encodeURIComponent(eventId)}/false-positive`,
        { false_positive: falsePositive, note },
      ),
  });
}
