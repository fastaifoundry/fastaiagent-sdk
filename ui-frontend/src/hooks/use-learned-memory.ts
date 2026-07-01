import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export interface LearnedFact {
  id: number;
  scope: string;
  scope_id: string;
  fact: string;
  source_trace_id: string | null;
  confidence: number;
  created_at: number;
  superseded_by: number | null;
  project_id: string;
}

export interface LearnedMemoryPage {
  rows: LearnedFact[];
  total: number;
  filters: {
    scope: string | null;
    scope_id: string | null;
    include_superseded: boolean;
  };
}

export interface MemoryScope {
  scope: string;
  scope_id: string;
  n: number;
}

export interface LearnedMemoryFilters {
  scope?: string | null;
  scope_id?: string | null;
  include_superseded?: boolean;
}

function buildQuery(filters: LearnedMemoryFilters, redact: boolean): string {
  const params = new URLSearchParams();
  if (filters.scope) params.set("scope", filters.scope);
  if (filters.scope_id) params.set("scope_id", filters.scope_id);
  if (filters.include_superseded) params.set("include_superseded", "true");
  // ``?redact=true`` is honored only when a read-mode RedactionPolicy is
  // installed; otherwise it's a no-op (mirrors useTrace).
  if (redact) params.set("redact", "true");
  const s = params.toString();
  return s ? `?${s}` : "";
}

export function useLearnedMemory(
  filters: LearnedMemoryFilters = {},
  redact: boolean = false
) {
  return useQuery({
    queryKey: ["learned-memory", filters, redact],
    queryFn: () =>
      api.get<LearnedMemoryPage>(`/learned_memory${buildQuery(filters, redact)}`),
  });
}

export function useLearnedMemoryScopes() {
  return useQuery({
    queryKey: ["learned-memory-scopes"],
    queryFn: () => api.get<{ scopes: MemoryScope[] }>("/learned_memory/scopes"),
  });
}
