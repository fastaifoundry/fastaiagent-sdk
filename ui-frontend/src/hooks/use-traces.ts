import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type {
  CompareTracesResponse,
  TraceDetail,
  TraceFilters,
  TracesPage,
  SpanTreeNode,
} from "@/lib/types";

function buildQuery(filters: TraceFilters): string {
  const params = new URLSearchParams();
  if (filters.agent) params.set("agent", filters.agent);
  if (filters.status) params.set("status", filters.status);
  if (filters.q) params.set("q", filters.q);
  if (filters.thread_id) params.set("thread_id", filters.thread_id);
  if (filters.runner_type) params.set("runner_type", filters.runner_type);
  if (filters.framework) params.set("framework", filters.framework);
  if (filters.since) params.set("since", filters.since);
  if (filters.until) params.set("until", filters.until);
  if (filters.min_duration_ms != null) params.set("min_duration_ms", String(filters.min_duration_ms));
  if (filters.max_duration_ms != null) params.set("max_duration_ms", String(filters.max_duration_ms));
  if (filters.min_cost != null) params.set("min_cost", String(filters.min_cost));
  if (filters.max_cost != null) params.set("max_cost", String(filters.max_cost));
  if (filters.min_tokens != null) params.set("min_tokens", String(filters.min_tokens));
  params.set("page", String(filters.page ?? 1));
  params.set("page_size", String(filters.page_size ?? 100));
  const s = params.toString();
  return s ? `?${s}` : "";
}

export function useTraces(filters: TraceFilters) {
  return useQuery({
    queryKey: ["traces", filters],
    queryFn: () => api.get<TracesPage>(`/traces${buildQuery(filters)}`),
  });
}

export function useTrace(traceId: string | undefined) {
  return useQuery({
    queryKey: ["trace", traceId],
    queryFn: () => api.get<TraceDetail>(`/traces/${traceId}`),
    enabled: !!traceId,
  });
}

export function useTraceSpans(traceId: string | undefined) {
  return useQuery({
    queryKey: ["trace-spans", traceId],
    queryFn: () => api.get<{ tree: SpanTreeNode }>(`/traces/${traceId}/spans`),
    enabled: !!traceId,
  });
}

export function useDeleteTrace() {
  return useMutation({
    mutationFn: (traceId: string) =>
      api.delete<{ deleted: number }>(`/traces/${traceId}`),
  });
}

export function useBulkDeleteTraces() {
  return useMutation({
    mutationFn: (traceIds: string[]) =>
      api.post<{ deleted: number; requested: number }>(
        "/traces/bulk-delete",
        { trace_ids: traceIds }
      ),
  });
}

export function useCompareTraces(
  a: string | null | undefined,
  b: string | null | undefined
) {
  return useQuery({
    queryKey: ["traces-compare", a, b],
    queryFn: () =>
      api.get<CompareTracesResponse>(
        `/traces/compare?a=${encodeURIComponent(a!)}&b=${encodeURIComponent(b!)}`
      ),
    enabled: !!a && !!b,
  });
}
