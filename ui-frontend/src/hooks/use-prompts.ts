import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type {
  PromptDetail,
  PromptLineage,
  PromptListResponse,
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
    queryFn: () =>
      api.get<{
        slug: string;
        version: number;
        template: string;
        variables: string[];
        metadata: Record<string, unknown>;
      }>(`/prompts/${slug}/versions/${version}`),
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
      metadata,
    }: {
      slug: string;
      template: string;
      metadata?: Record<string, unknown>;
    }) =>
      api.put<{ slug: string; version: number; template: string }>(
        `/prompts/${slug}`,
        { template, metadata: metadata ?? {} }
      ),
  });
}
