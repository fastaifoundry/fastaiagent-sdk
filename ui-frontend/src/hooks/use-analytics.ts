import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useIncludeExperiments } from "@/stores/preferences-store";
import type { AnalyticsPayload, TraceScores, ThreadDetail } from "@/lib/types";

export function useAnalytics(windowHours: number = 168, granularity: "hour" | "day" = "hour") {
  // Experiments (replays, Playground, eval…) are left out unless the user
  // opted in; the payload reports them apart either way (1.89.0).
  const includeExperiments = useIncludeExperiments();
  return useQuery({
    queryKey: ["analytics", windowHours, granularity, includeExperiments],
    queryFn: () =>
      api.get<AnalyticsPayload>(
        `/analytics?hours=${windowHours}&granularity=${granularity}` +
          `&include_experiments=${includeExperiments}`
      ),
  });
}

export function useTraceScores(traceId: string | undefined) {
  return useQuery({
    queryKey: ["trace-scores", traceId],
    queryFn: () => api.get<TraceScores>(`/traces/${traceId}/scores`),
    enabled: !!traceId,
  });
}

export function useThread(threadId: string | undefined) {
  return useQuery({
    queryKey: ["thread", threadId],
    queryFn: () => api.get<ThreadDetail>(`/threads/${threadId}`),
    enabled: !!threadId,
  });
}
