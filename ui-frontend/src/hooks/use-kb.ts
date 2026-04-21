import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type {
  KbChunksResponse,
  KbDetail,
  KbDocumentsResponse,
  KbLineageResponse,
  KbListResponse,
  KbSearchResponse,
} from "@/lib/types";

export function useKbCollections() {
  return useQuery({
    queryKey: ["kb", "list"],
    queryFn: () => api.get<KbListResponse>("/kb"),
  });
}

export function useKbDetail(name: string | undefined) {
  return useQuery({
    queryKey: ["kb", name, "detail"],
    queryFn: () => api.get<KbDetail>(`/kb/${encodeURIComponent(name!)}`),
    enabled: !!name,
  });
}

export function useKbDocuments(
  name: string | undefined,
  page = 1,
  pageSize = 50
) {
  return useQuery({
    queryKey: ["kb", name, "documents", page, pageSize],
    queryFn: () =>
      api.get<KbDocumentsResponse>(
        `/kb/${encodeURIComponent(name!)}/documents?page=${page}&page_size=${pageSize}`
      ),
    enabled: !!name,
  });
}

export function useKbChunks(name: string | undefined, source: string | null) {
  return useQuery({
    queryKey: ["kb", name, "chunks", source],
    queryFn: () =>
      api.get<KbChunksResponse>(
        `/kb/${encodeURIComponent(name!)}/chunks?source=${encodeURIComponent(
          source!
        )}`
      ),
    enabled: !!name && !!source,
  });
}

export function useKbLineage(name: string | undefined) {
  return useQuery({
    queryKey: ["kb", name, "lineage"],
    queryFn: () =>
      api.get<KbLineageResponse>(`/kb/${encodeURIComponent(name!)}/lineage`),
    enabled: !!name,
  });
}

export function useKbSearch(name: string | undefined) {
  return useMutation({
    mutationKey: ["kb", name, "search"],
    mutationFn: (payload: { query: string; top_k: number }) =>
      api.post<KbSearchResponse>(
        `/kb/${encodeURIComponent(name!)}/search`,
        payload
      ),
  });
}
