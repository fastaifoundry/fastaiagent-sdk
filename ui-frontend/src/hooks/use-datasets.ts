import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type {
  CaseBody,
  DatasetCase,
  DatasetDetail,
  DatasetImageUploadResult,
  DatasetImportResult,
  DatasetRunEvalResult,
  DatasetSummary,
} from "@/lib/types";

/**
 * React Query hooks for the Eval Dataset Editor (Sprint 3).
 *
 * Mutations invalidate the matching list/detail queries so the UI
 * stays consistent without manual refetches.
 */

export const datasetKeys = {
  all: ["datasets"] as const,
  list: () => ["datasets", "list"] as const,
  detail: (name: string) => ["datasets", "detail", name] as const,
};

export function useDatasets() {
  return useQuery({
    queryKey: datasetKeys.list(),
    queryFn: () => api.get<DatasetSummary[]>("/datasets"),
  });
}

export function useDataset(name: string | undefined | null) {
  return useQuery({
    queryKey: datasetKeys.detail(name ?? ""),
    queryFn: () => api.get<DatasetDetail>(`/datasets/${encodeURIComponent(name!)}`),
    enabled: !!name,
  });
}

export function useCreateDataset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) =>
      api.post<DatasetSummary>("/datasets", { name }),
    onSuccess: () => qc.invalidateQueries({ queryKey: datasetKeys.list() }),
  });
}

export function useDeleteDataset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) =>
      api.delete<void>(`/datasets/${encodeURIComponent(name)}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: datasetKeys.list() }),
  });
}

export function useAddCase(name: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: CaseBody) =>
      api.post<DatasetCase>(`/datasets/${encodeURIComponent(name)}/cases`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: datasetKeys.detail(name) });
      qc.invalidateQueries({ queryKey: datasetKeys.list() });
    },
  });
}

export function useUpdateCase(name: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ index, body }: { index: number; body: CaseBody }) =>
      api.put<DatasetCase>(
        `/datasets/${encodeURIComponent(name)}/cases/${index}`,
        body
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: datasetKeys.detail(name) });
    },
  });
}

export function useDeleteCase(name: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (index: number) =>
      api.delete<void>(
        `/datasets/${encodeURIComponent(name)}/cases/${index}`
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: datasetKeys.detail(name) });
      qc.invalidateQueries({ queryKey: datasetKeys.list() });
    },
  });
}

/**
 * Multipart-friendly mutation hooks: these bypass ``api.ts`` because
 * its JSON wrapper insists on Content-Type: application/json. The
 * fetch call here is otherwise identical (same credentials, same
 * relative URL).
 */
async function postFormData<T>(path: string, fd: FormData): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method: "POST",
    body: fd,
    credentials: "include",
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = (body as { detail?: string }).detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export function useUploadImage(name: string) {
  return useMutation({
    mutationFn: (file: File) => {
      const fd = new FormData();
      fd.append("file", file);
      return postFormData<DatasetImageUploadResult>(
        `/datasets/${encodeURIComponent(name)}/images`,
        fd
      );
    },
  });
}

export function useImportJsonl(name: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      file,
      mode,
    }: {
      file: File;
      mode: "append" | "replace";
    }) => {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("mode", mode);
      return postFormData<DatasetImportResult>(
        `/datasets/${encodeURIComponent(name)}/import`,
        fd
      );
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: datasetKeys.detail(name) });
      qc.invalidateQueries({ queryKey: datasetKeys.list() });
    },
  });
}

export function useRunEval(name: string) {
  return useMutation({
    mutationFn: (body: {
      agent_name?: string | null;
      scorers?: string[];
      run_name?: string;
    }) =>
      api.post<DatasetRunEvalResult>(
        `/datasets/${encodeURIComponent(name)}/run-eval`,
        body
      ),
  });
}
