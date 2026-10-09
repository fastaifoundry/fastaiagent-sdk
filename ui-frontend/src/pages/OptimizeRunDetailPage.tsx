import { useState } from "react";
import { Check, Copy, ExternalLink, RefreshCw } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import { PageHeader } from "@/components/layout/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { useOptimizeRun } from "@/hooks/use-optimizes";
import type { OptimizeIterationRow, OptimizeRunRow } from "@/lib/types";

function fmtScore(n: number | null | undefined): string {
  return n != null ? n.toFixed(3) : "—";
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="space-y-0.5">
      <div className="text-xs uppercase text-muted-foreground">{label}</div>
      <div className="font-mono text-sm tabular-nums">{value}</div>
    </div>
  );
}

function statusBadge(it: OptimizeIterationRow) {
  if (it.skipped) return <Badge variant="outline">skipped</Badge>;
  if (it.accepted) return <Badge variant="default">accepted</Badge>;
  return <Badge variant="secondary">rejected</Badge>;
}

function asText(v: unknown): string {
  return typeof v === "string" ? v : JSON.stringify(v);
}

function PromptBlock({ label, text }: { label: string; text: string }) {
  const [copied, setCopied] = useState(false);

  function handleCopy() {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase text-muted-foreground">{label}</div>
        <button
          type="button"
          onClick={handleCopy}
          title={`Copy ${label.toLowerCase()}`}
          className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-md border bg-muted/40 p-3 font-mono text-xs">
        {text}
      </pre>
    </div>
  );
}

/** What the run would ship: the winning prompt next to the one it started from,
 *  plus the few-shot examples and learned facts it selected. */
function WinnerCard({ run }: { run: OptimizeRunRow }) {
  const cand = run.best_candidate ?? {};
  const meta = run.metadata ?? {};
  const winning = typeof cand.system_prompt === "string" ? cand.system_prompt : null;
  const original =
    typeof meta.baseline_system_prompt === "string" ? meta.baseline_system_prompt : null;
  const demos = Array.isArray(cand.fewshot_demos) ? cand.fewshot_demos : [];
  const facts = Array.isArray(cand.fact_ids) ? cand.fact_ids : [];
  const errors = Array.isArray(meta.proposer_errors) ? meta.proposer_errors : [];
  const changed = winning !== null || demos.length > 0 || facts.length > 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Winner</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {run.reverted ? (
          <p className="text-sm text-muted-foreground">
            The winner regressed on the holdout and was reverted — the agent keeps its
            original configuration.
          </p>
        ) : !changed ? (
          <p className="text-sm text-muted-foreground">
            No candidate beat the baseline — the agent keeps its original configuration.
          </p>
        ) : null}

        {winning !== null ? (
          <>
            <PromptBlock label="Winning prompt" text={winning} />
            {original !== null ? <PromptBlock label="Original prompt" text={original} /> : null}
          </>
        ) : original !== null ? (
          <PromptBlock label="Prompt (unchanged)" text={original} />
        ) : (
          <p className="text-xs text-muted-foreground">System prompt: unchanged.</p>
        )}

        {demos.length > 0 ? (
          <details className="text-xs">
            <summary className="cursor-pointer text-muted-foreground">
              {demos.length} few-shot example{demos.length === 1 ? "" : "s"}
            </summary>
            <ul className="mt-2 space-y-1.5">
              {demos.map((d, i) => (
                <li key={i} className="rounded-md border bg-muted/40 p-2 font-mono">
                  <div>
                    <span className="text-muted-foreground">input: </span>
                    {asText(d.input)}
                  </div>
                  <div>
                    <span className="text-muted-foreground">output: </span>
                    {asText(d.output)}
                  </div>
                </li>
              ))}
            </ul>
          </details>
        ) : null}

        {facts.length > 0 ? (
          <div className="text-xs text-muted-foreground">
            Learned facts injected: {facts.map((id) => `#${id}`).join(", ")}
          </div>
        ) : null}

        {errors.length > 0 ? (
          <div className="rounded-md border border-destructive/40 p-2 text-xs">
            The prompt proposer failed {errors.length} time{errors.length === 1 ? "" : "s"}:{" "}
            <span className="font-mono">{errors[errors.length - 1]}</span>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

export function OptimizeRunDetailPage() {
  const { runId } = useParams<{ runId: string }>();
  const query = useOptimizeRun(runId);
  const run = query.data?.run;
  const iterations = query.data?.iterations ?? [];

  const baseline = run?.baseline_score ?? null;
  const best = run?.best_score ?? null;
  const delta =
    baseline != null && best != null ? best - baseline : null;

  return (
    <div className="space-y-5">
      <PageHeader
        title={run?.run_name || run?.run_id || "AutoLLM run"}
        description={run?.agent_name ? `agent: ${run.agent_name}` : undefined}
      >
        <Button
          variant="outline"
          size="sm"
          onClick={() => query.refetch()}
          disabled={query.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${query.isFetching ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </PageHeader>

      {query.isLoading ? (
        <TableSkeleton rows={6} />
      ) : !run ? (
        <EmptyState title="Run not found" description="This optimize run does not exist." />
      ) : (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-6">
                <Stat label="baseline (dev)" value={fmtScore(baseline)} />
                <Stat label="best (dev)" value={fmtScore(best)} />
                <Stat
                  label="Δ dev"
                  value={delta != null ? `${delta >= 0 ? "+" : ""}${delta.toFixed(3)}` : "—"}
                />
                <Stat label="holdout best" value={fmtScore(run.holdout_best_score)} />
                <Stat label="seed" value={String(run.seed ?? "—")} />
                <div className="space-y-0.5">
                  <div className="text-xs uppercase text-muted-foreground">outcome</div>
                  <div>
                    {run.reverted ? (
                      <Badge variant="destructive">reverted</Badge>
                    ) : (
                      <Badge variant="secondary">{run.stopped_reason || "—"}</Badge>
                    )}
                  </div>
                </div>
              </div>
              <div className="mt-4 text-xs text-muted-foreground">
                levers: {run.levers?.length ? run.levers.join(" → ") : "—"}
              </div>
            </CardContent>
          </Card>

          <WinnerCard run={run} />

          {iterations.length === 0 ? (
            <EmptyState title="No trajectory" description="This run recorded no iterations." />
          ) : (
            <div className="rounded-md border bg-card">
              <Table>
                <TableHeader>
                  <TableRow className="hover:bg-transparent">
                    <TableHead className="w-[60px]">Iter</TableHead>
                    <TableHead>Lever</TableHead>
                    <TableHead className="w-[100px]">Dev score</TableHead>
                    <TableHead className="w-[120px]">Status</TableHead>
                    <TableHead>Rationale</TableHead>
                    <TableHead className="w-[90px]">Eval</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {iterations.map((it) => (
                    <TableRow key={it.iteration_id} className="hover:bg-muted/40">
                      <TableCell className="font-mono text-xs tabular-nums">
                        {it.iteration}
                      </TableCell>
                      <TableCell className="text-sm">{it.lever}</TableCell>
                      <TableCell className="font-mono text-xs tabular-nums">
                        {fmtScore(it.dev_score)}
                      </TableCell>
                      <TableCell>{statusBadge(it)}</TableCell>
                      <TableCell className="max-w-[420px] truncate text-xs text-muted-foreground">
                        {it.rationale || "—"}
                      </TableCell>
                      <TableCell>
                        {it.eval_run_id ? (
                          <Link
                            to={`/evals/${it.eval_run_id}`}
                            title="Open the eval run that scored this candidate"
                            className="inline-flex items-center text-muted-foreground hover:text-primary"
                          >
                            <ExternalLink className="h-3.5 w-3.5" />
                          </Link>
                        ) : (
                          "—"
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
