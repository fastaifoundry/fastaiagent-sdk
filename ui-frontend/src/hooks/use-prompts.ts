import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type {
  PromptDetail,
  PromptLineage,
  PromptListResponse,
  PromptVersionDetail,
  PromptVersionRow,
} from "@/lib/types";

export function usePrompts() {
  return useQuery({
    queryKey: ["prompts"],
    queryFn: () => api.get<PromptListResponse>("/prompts"),
  });
}

export function usePrompt(slug: string | undefined) {
  return useQuery({
    queryKey: ["prompt", slug],
    queryFn: () => api.get<PromptDetail>(`/prompts/${slug}`),
    enabled: !!slug,
  });
}

export function usePromptVersions(slug: string | undefined) {
  return useQuery({
    queryKey: ["prompt-versions", slug],
    queryFn: () =>
      api.get<{ versions: PromptVersionRow[] }>(`/prompts/${slug}/versions`),
    enabled: !!slug,
  });
}

export function usePromptVersion(slug: string | undefined, version: string | null) {
  return useQuery({
    queryKey: ["prompt-version", slug, version],
    queryFn: () => api.get<PromptVersionDetail>(`/prompts/${slug}/versions/${version}`),
    enabled: !!slug && !!version,
  });
}

export function usePromptLineage(slug: string | undefined) {
  return useQuery({
    queryKey: ["prompt-lineage", slug],
    queryFn: () => api.get<PromptLineage>(`/prompts/${slug}/lineage`),
    enabled: !!slug,
  });
}

export function useUpdatePrompt() {
  return useMutation({
    mutationFn: ({
      slug,
      template,
      baseVersion,
      metadata,
    }: {
      slug: string;
      template: string;
      /** The version the edit started from; the server copies its metadata. */
      baseVersion?: number;
      /** Only when the user changed it: omitted means "inherit" (1.89.0). */
      metadata?: Record<string, unknown>;
    }) =>
      api.put<PromptVersionDetail>(`/prompts/${slug}`, {
        template,
        ...(baseVersion !== undefined ? { base_version: baseVersion } : {}),
        ...(metadata !== undefined ? { metadata } : {}),
      }),
  });
}

export function useDeletePrompt() {
  return useMutation({
    mutationFn: ({ slug }: { slug: string }) =>
      api.delete<{ slug: string; versions_deleted: number }>(
        `/prompts/${encodeURIComponent(slug)}`
      ),
  });
}
