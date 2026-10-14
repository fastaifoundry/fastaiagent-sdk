import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError, readCsrfCookie } from "@/lib/api";
import { normalizeErrorBody } from "@/lib/api-error";
import type {
  CaseBody,
  DatasetCase,
  DatasetDetail,
  DatasetImageUploadResult,
  DatasetImportResult,
  DatasetRunEvalResult,
  DatasetSummary,
  RunEvalBody,
  RunEvalOptions,
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
  // Echo the CSRF double-submit token like api.ts does — the middleware
  // requires it on state-changing requests under auth (security_audit_2 M-10).
  // No Content-Type: the browser sets the multipart boundary for FormData.
  const headers: Record<string, string> = {};
  const csrf = readCsrfCookie();
  if (csrf) headers["X-CSRF-Token"] = csrf;
  const res = await fetch(`/api${path}`, {
    method: "POST",
    body: fd,
    credentials: "include",
    headers,
  });
  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      /* ignore */
    }
    const info = normalizeErrorBody(body, res.status, res.statusText);
    throw new ApiError(res.status, info.message, body, info);
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

/**
 * What the Run eval dialog can offer for this dataset: registered agents,
 * the scorer catalog with per-dataset availability, judge models with a key,
 * and what the rows carry. Fetched only while the dialog is open.
 */
export function useRunEvalOptions(name: string, enabled: boolean) {
  return useQuery({
    queryKey: [...datasetKeys.detail(name), "run-eval-options"] as const,
    queryFn: () =>
      api.get<RunEvalOptions>(
        `/datasets/${encodeURIComponent(name)}/run-eval/options`
      ),
    enabled: enabled && !!name,
  });
}

/** Runs synchronously on the server: the request lasts as long as the eval. */
export function useRunEval(name: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: RunEvalBody) =>
      api.post<DatasetRunEvalResult>(
        `/datasets/${encodeURIComponent(name)}/run-eval`,
        body
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["evals"] });
      qc.invalidateQueries({ queryKey: ["eval-trend"] });
    },
  });
}
