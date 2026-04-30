import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export interface PendingInterrupt {
  execution_id: string;
  chain_name: string;
  node_id: string;
  reason: string;
  context: Record<string, unknown>;
  agent_path: string | null;
  created_at: string;
}

interface PendingPayload {
  count: number;
  items: PendingInterrupt[];
}

/** Polls only on focus / mount — refresh-based reads (memory rule). */
export function usePendingInterrupts(limit = 100) {
  return useQuery({
    queryKey: ["pending-interrupts", limit],
    queryFn: () =>
      api.get<PendingPayload>(`/pending-interrupts?limit=${limit}`),
  });
}

export interface ResumeBody {
  approved: boolean;
  metadata?: Record<string, unknown>;
  reason?: string;
}

interface ResumeResponse {
  execution_id: string;
  chain_name: string;
  reason: string | null;
  result: {
    status: "completed" | "paused";
    output?: string;
    pending_interrupt?: Record<string, unknown> | null;
    [key: string]: unknown;
  };
}

export function useResumeExecution(executionId: string) {
  return useMutation({
    mutationFn: (body: ResumeBody) =>
      api.post<ResumeResponse>(`/executions/${executionId}/resume`, body),
  });
}
