import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export interface IdempotencyCacheRow {
  function_key: string;
  result: unknown;
  created_at: string;
}

interface IdempotencyCacheResponse {
  execution_id: string;
  count: number;
  items: IdempotencyCacheRow[];
}

export function useIdempotencyCache(executionId: string | undefined) {
  return useQuery({
    queryKey: ["idempotency-cache", executionId],
    queryFn: () =>
      api.get<IdempotencyCacheResponse>(
        `/executions/${executionId}/idempotency-cache`
      ),
    enabled: !!executionId,
  });
}
