import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type {
  ComparisonResult,
  ForkModifications,
  RerunResult,
  ReplayStep,
} from "@/lib/types";

export function useReplay(traceId: string | undefined) {
  return useQuery({
    queryKey: ["replay", traceId],
    queryFn: () =>
      api.get<{ steps: ReplayStep[] }>(`/replay/${traceId}`),
    enabled: !!traceId,
  });
}

export function useForkAtStep(traceId: string | undefined) {
  return useMutation({
    mutationFn: (step: number) =>
      api.post<{ fork_id: string }>(`/replay/${traceId}/fork`, { step }),
  });
}

export function useModifyFork() {
  return useMutation({
    mutationFn: ({
      forkId,
      mods,
    }: {
      forkId: string;
      mods: ForkModifications;
    }) => api.patch<{ fork_id: string; status: string }>(`/replay/forks/${forkId}`, mods),
  });
}

export function useRerunFork() {
  return useMutation({
    mutationFn: (forkId: string) =>
      api.post<RerunResult>(`/replay/forks/${forkId}/rerun`),
  });
}

export function useCompareFork() {
  return useMutation({
    mutationFn: ({
      forkId,
      against,
    }: {
      forkId: string;
      against: string;
    }) =>
      api.get<ComparisonResult>(
        `/replay/forks/${forkId}/compare?against=${encodeURIComponent(against)}`
      ),
  });
}

export function useSaveAsTest() {
  return useMutation({
    mutationFn: ({
      forkId,
      input,
      expectedOutput,
    }: {
      forkId: string;
      input: string;
      expectedOutput: string;
    }) =>
      api.post<{ path: string }>(`/replay/forks/${forkId}/save-as-test`, {
        input,
        expected_output: expectedOutput,
      }),
  });
}
