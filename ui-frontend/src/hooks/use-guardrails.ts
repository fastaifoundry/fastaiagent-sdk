import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { GuardrailEventsPage } from "@/lib/types";

interface Filters {
  rule?: string | null;
  outcome?: string | null;
  agent?: string | null;
  since?: string;
  until?: string;
  page?: number;
  page_size?: number;
}

function buildQuery(f: Filters): string {
  const params = new URLSearchParams();
  if (f.rule) params.set("rule", f.rule);
  if (f.outcome) params.set("outcome", f.outcome);
  if (f.agent) params.set("agent", f.agent);
  if (f.since) params.set("since", f.since);
  if (f.until) params.set("until", f.until);
  params.set("page", String(f.page ?? 1));
  params.set("page_size", String(f.page_size ?? 100));
  return params.toString() ? `?${params.toString()}` : "";
}

export function useGuardrailEvents(filters: Filters = {}) {
  return useQuery({
    queryKey: ["guardrails", filters],
    queryFn: () =>
      api.get<GuardrailEventsPage>(`/guardrail-events${buildQuery(filters)}`),
  });
}
