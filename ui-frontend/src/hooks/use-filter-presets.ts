import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { FilterPreset, TraceFilters } from "@/lib/types";

/**
 * React Query hooks for the saved filter presets endpoints
 * (``/api/filter-presets``). All mutations invalidate the list query
 * so the dropdown updates without manual refetch.
 */

const KEY = ["filter-presets"] as const;

export function useFilterPresets() {
  return useQuery({
    queryKey: KEY,
    queryFn: () => api.get<FilterPreset[]>("/filter-presets"),
  });
}

export function useCreateFilterPreset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ name, filters }: { name: string; filters: TraceFilters }) =>
      api.post<FilterPreset>("/filter-presets", { name, filters }),
    onSuccess: () => qc.invalidateQueries({ queryKey: KEY }),
  });
}

export function useUpdateFilterPreset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      name,
      filters,
    }: {
      id: string;
      name?: string;
      filters?: TraceFilters;
    }) => api.patch<FilterPreset>(`/filter-presets/${id}`, { name, filters }),
    onSuccess: () => qc.invalidateQueries({ queryKey: KEY }),
  });
}

export function useDeleteFilterPreset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.delete<void>(`/filter-presets/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: KEY }),
  });
}
