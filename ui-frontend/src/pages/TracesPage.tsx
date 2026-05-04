import { useCallback, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/shared/EmptyState";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { TraceFiltersBar } from "@/components/traces/TraceFilters";
import { TracesTable } from "@/components/traces/TracesTable";
import { useTraces } from "@/hooks/use-traces";
import type { RunnerType, TraceFilters } from "@/lib/types";

const NUM_KEYS: (keyof TraceFilters)[] = [
  "min_duration_ms",
  "max_duration_ms",
  "min_cost",
  "max_cost",
  "min_tokens",
  "page",
  "page_size",
];

const STR_KEYS: (keyof TraceFilters)[] = [
  "agent",
  "status",
  "q",
  "thread_id",
  "runner_name",
  "since",
  "until",
];

const VALID_RUNNERS: RunnerType[] = ["agent", "chain", "swarm", "supervisor"];
// Backend regex for ``framework`` query param: leading letter then
// letters / digits / [_.-]. Mirroring it on the client lets us reject
// pathological URL params before they round-trip to the API.
const FRAMEWORK_PATTERN = /^[A-Za-z][A-Za-z0-9_.-]{0,63}$/;

function paramsToFilters(sp: URLSearchParams): TraceFilters {
  const out: TraceFilters = { page: 1, page_size: 100 };
  for (const key of STR_KEYS) {
    const v = sp.get(key);
    if (v != null && v !== "") {
      // The cast is safe: keys in STR_KEYS only point at string-typed
      // fields on TraceFilters.
      (out as Record<string, unknown>)[key] = v;
    }
  }
  for (const key of NUM_KEYS) {
    const v = sp.get(key);
    if (v != null && v !== "") {
      const n = Number(v);
      if (Number.isFinite(n)) {
        (out as Record<string, unknown>)[key] = n;
      }
    }
  }
  const rt = sp.get("runner_type");
  if (rt && (VALID_RUNNERS as string[]).includes(rt)) {
    out.runner_type = rt as RunnerType;
  }
  const fw = sp.get("framework");
  if (fw && FRAMEWORK_PATTERN.test(fw)) {
    out.framework = fw;
  }
  return out;
}

function filtersToParams(filters: TraceFilters): URLSearchParams {
  const sp = new URLSearchParams();
  for (const key of [...STR_KEYS, "runner_type", "framework"] as const) {
    const v = (filters as Record<string, unknown>)[key];
    if (typeof v === "string" && v) sp.set(key, v);
  }
  for (const key of NUM_KEYS) {
    if (key === "page" || key === "page_size") continue;
    const v = (filters as Record<string, unknown>)[key];
    if (typeof v === "number" && Number.isFinite(v)) sp.set(key, String(v));
  }
  return sp;
}

/**
 * Traces list page. Filter state is lifted into the URL query string
 * (Sprint 3) so refresh, back/forward, bookmark, and share-via-link
 * all preserve the active filters. ``useSearchParams`` is the source
 * of truth; ``TraceFiltersBar`` reads + writes through ``onChange``.
 */
export function TracesPage() {
  const [params, setParams] = useSearchParams();

  const filters = useMemo(() => paramsToFilters(params), [params]);

  const setFilters = useCallback(
    (next: TraceFilters) => {
      const sp = filtersToParams(next);
      // ``replace: true`` so the URL doesn't accumulate history entries
      // for every keystroke / filter tweak.
      setParams(sp, { replace: true });
    },
    [setParams]
  );

  const { data, isLoading, isFetching, refetch } = useTraces(filters);

  const rows = data?.rows ?? [];

  return (
    <div className="space-y-5">
      <PageHeader
        title="Traces"
        description={
          data
            ? `${data.total.toLocaleString()} trace${data.total === 1 ? "" : "s"}`
            : undefined
        }
      >
        <Button
          variant="outline"
          size="sm"
          onClick={() => refetch()}
          disabled={isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${isFetching ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </PageHeader>

      <TraceFiltersBar filters={filters} onChange={setFilters} />

      {isLoading ? (
        <TableSkeleton rows={10} />
      ) : rows.length === 0 ? (
        <EmptyState
          title="No traces match these filters"
          description="Try widening the time range or clearing the search term."
        />
      ) : (
        <TracesTable rows={rows} />
      )}
    </div>
  );
}
