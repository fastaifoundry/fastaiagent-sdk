import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  Activity,
  ChevronLeft,
  Database,
  FileText,
  Play,
  RefreshCw,
  Search,
} from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { StatCard } from "@/components/shared/StatCard";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { JsonViewer } from "@/components/shared/JsonViewer";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  useKbChunks,
  useKbDetail,
  useKbDocuments,
  useKbLineage,
  useKbSearch,
} from "@/hooks/use-kb";
import { formatTimeAgo } from "@/lib/format";
import type { KbSearchHit } from "@/lib/types";

export function KbDetailPage() {
  const { name } = useParams<{ name: string }>();
  const detail = useKbDetail(name);
  const docs = useKbDocuments(name, 1, 100);

  if (detail.isLoading) return <TableSkeleton rows={4} />;
  if (detail.error || !detail.data) {
    return <EmptyState title="KB not found" icon={Database} />;
  }
  const data = detail.data;

  return (
    <div className="space-y-5">
      <PageHeader title={data.name} description={data.path}>
        <Link to="/kb">
          <Button variant="ghost" size="sm">
            <ChevronLeft className="mr-1.5 h-3.5 w-3.5" />
            Back
          </Button>
        </Link>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            detail.refetch();
            docs.refetch();
          }}
          disabled={detail.isFetching || docs.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${
              detail.isFetching || docs.isFetching ? "animate-spin" : ""
            }`}
          />
          Refresh
        </Button>
      </PageHeader>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard label="Documents" value={String(data.doc_count)} />
        <StatCard label="Chunks" value={String(data.chunk_count)} />
        <StatCard label="Size" value={formatBytes(data.size_bytes)} />
        <StatCard label="Updated" value={formatTimeAgo(data.last_updated)} />
      </div>

      <Tabs defaultValue="documents">
        <TabsList>
          <TabsTrigger value="documents">
            <FileText className="mr-1.5 h-3.5 w-3.5" />
            Documents
          </TabsTrigger>
          <TabsTrigger value="search">
            <Search className="mr-1.5 h-3.5 w-3.5" />
            Search playground
          </TabsTrigger>
          <TabsTrigger value="lineage">
            <Activity className="mr-1.5 h-3.5 w-3.5" />
            Lineage
          </TabsTrigger>
        </TabsList>

        <TabsContent value="documents">
          <DocumentsTab name={name!} />
        </TabsContent>
        <TabsContent value="search">
          <SearchTab name={name!} />
        </TabsContent>
        <TabsContent value="lineage">
          <LineageTab name={name!} />
        </TabsContent>
      </Tabs>
    </div>
  );
}

function DocumentsTab({ name }: { name: string }) {
  const docs = useKbDocuments(name, 1, 100);
  const [openSource, setOpenSource] = useState<string | null>(null);
  const chunks = useKbChunks(name, openSource);
  const rows = docs.data?.documents ?? [];

  if (docs.isLoading) return <TableSkeleton rows={6} />;
  if (rows.length === 0) {
    return (
      <EmptyState
        title="No documents indexed"
        icon={FileText}
        description="Run kb.add() in code, then refresh."
      />
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-[1fr_1fr]">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">
            {docs.data?.total.toLocaleString()} document
            {docs.data?.total === 1 ? "" : "s"}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 pt-0">
          {rows.map((doc) => (
            <button
              key={doc.source}
              onClick={() => setOpenSource(doc.source)}
              className={`block w-full rounded-md border px-3 py-2 text-left transition-colors hover:border-primary ${
                openSource === doc.source
                  ? "border-primary bg-muted/40"
                  : "border-transparent"
              }`}
            >
              <div className="flex items-center justify-between gap-3">
                <span className="truncate font-mono text-xs">{doc.source}</span>
                <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 text-[10px] font-mono tabular-nums">
                  {doc.chunk_count}
                </span>
              </div>
              <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">
                {doc.preview}
              </p>
            </button>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">
            {openSource ? "Chunks" : "Pick a document to inspect"}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 pt-0">
          {!openSource ? (
            <p className="py-6 text-center text-xs text-muted-foreground">
              Left pane lists all documents ingested into this KB. Click one to
              see its chunks.
            </p>
          ) : chunks.isLoading ? (
            <TableSkeleton rows={4} />
          ) : (
            (chunks.data?.chunks ?? []).map((c) => (
              <div key={c.id} className="rounded-md border p-3 text-xs">
                <div className="mb-2 flex items-center justify-between gap-2 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
                  <span>Chunk {c.index}</span>
                  <span>
                    {c.start_char} – {c.end_char}
                  </span>
                </div>
                <p className="whitespace-pre-wrap">{c.content}</p>
              </div>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function SearchTab({ name }: { name: string }) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const search = useKbSearch(name);

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    search.mutate({ query, top_k: topK });
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm">Query this KB</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <form onSubmit={onSubmit} className="flex flex-col gap-3 sm:flex-row sm:items-end">
          <div className="flex-1">
            <Label htmlFor="query" className="mb-1 text-xs">
              Query
            </Label>
            <Input
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="refund policy"
              autoFocus
            />
          </div>
          <div className="w-24">
            <Label htmlFor="top_k" className="mb-1 text-xs">
              Top-k
            </Label>
            <Input
              id="top_k"
              type="number"
              min={1}
              max={50}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
          </div>
          <Button type="submit" disabled={search.isPending || !query.trim()}>
            <Play className="mr-1.5 h-3.5 w-3.5" />
            {search.isPending ? "Searching…" : "Run"}
          </Button>
        </form>

        {search.isError && (
          <p className="text-xs text-destructive">
            {search.error instanceof Error ? search.error.message : "Search failed"}
          </p>
        )}

        {search.data && (
          <div className="space-y-3">
            <p className="text-xs text-muted-foreground">
              {search.data.results.length} result
              {search.data.results.length === 1 ? "" : "s"} · search_type ={" "}
              <span className="font-mono">{search.data.search_type}</span>
            </p>
            {search.data.results.length === 0 ? (
              <EmptyState
                title="No matches"
                description="Try a broader query or lower the chunk_size."
              />
            ) : (
              search.data.results.map((r, i) => <ResultCard key={r.id} rank={i + 1} hit={r} />)
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ResultCard({ rank, hit }: { rank: number; hit: KbSearchHit }) {
  return (
    <div className="rounded-md border p-3 text-xs">
      <div className="mb-2 flex items-center justify-between gap-2 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        <span>
          #{rank} · {hit.source ?? hit.id}
        </span>
        <span>score {(hit.score ?? 0).toFixed(3)}</span>
      </div>
      <p className="whitespace-pre-wrap mb-2">{hit.content}</p>
      {Object.keys(hit.metadata ?? {}).length > 0 && (
        <details>
          <summary className="cursor-pointer text-[10px] text-muted-foreground">
            metadata
          </summary>
          <div className="mt-2">
            <JsonViewer data={hit.metadata} collapsed />
          </div>
        </details>
      )}
    </div>
  );
}

function LineageTab({ name }: { name: string }) {
  const lineage = useKbLineage(name);

  if (lineage.isLoading) return <TableSkeleton rows={4} />;
  if (!lineage.data) return null;
  const { retrieval_count, agents, recent_traces } = lineage.data;

  if (retrieval_count === 0) {
    return (
      <EmptyState
        title="No retrievals recorded yet"
        icon={Activity}
        description="Wire this KB into an agent with kb.as_tool() and run it — retrieval spans will land here."
      />
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">
            {retrieval_count.toLocaleString()} retrieval
            {retrieval_count === 1 ? "" : "s"} across {agents.length} agent
            {agents.length === 1 ? "" : "s"}
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          {agents.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              Retrievals were recorded but no agent attribution was attached.
            </p>
          ) : (
            <ul className="space-y-1.5">
              {agents.map((a) => (
                <li
                  key={a.agent_name}
                  className="flex items-center justify-between text-xs"
                >
                  <Link
                    to={`/agents/${encodeURIComponent(a.agent_name)}`}
                    className="font-mono hover:text-primary"
                  >
                    {a.agent_name}
                  </Link>
                  <span className="rounded bg-muted px-1.5 py-0.5 font-mono tabular-nums">
                    {a.retrieval_count}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Recent traces</CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          {recent_traces.length === 0 ? (
            <p className="text-xs text-muted-foreground">None yet.</p>
          ) : (
            <ul className="space-y-1.5 text-xs">
              {recent_traces.map((t) => (
                <li key={t.trace_id} className="flex items-center justify-between gap-2">
                  <Link
                    to={`/traces/${t.trace_id}`}
                    className="truncate font-mono hover:text-primary"
                  >
                    {t.name}
                  </Link>
                  <span className="shrink-0 text-muted-foreground">
                    {formatTimeAgo(t.start_time)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}
