import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  ChevronLeft,
  ExternalLink,
  Loader2,
  RefreshCw,
  Save,
} from "lucide-react";
import { toast } from "sonner";
import { useQueryClient } from "@tanstack/react-query";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { RegistryExternalBanner } from "@/components/prompts/RegistryGate";
import {
  usePrompt,
  usePromptLineage,
  usePromptVersion,
  usePromptVersions,
  useUpdatePrompt,
} from "@/hooks/use-prompts";
import { ApiError } from "@/lib/api";
import { cn } from "@/lib/utils";
import { formatTimeAgo } from "@/lib/format";

function extractVars(template: string): string[] {
  const matches = template.matchAll(/\{\{(\w+)\}\}/g);
  return Array.from(new Set(Array.from(matches, (m) => m[1])));
}

export function PromptEditorPage() {
  const { slug } = useParams<{ slug: string }>();
  const queryClient = useQueryClient();

  const prompt = usePrompt(slug);
  const versions = usePromptVersions(slug);
  const lineage = usePromptLineage(slug);
  const updatePrompt = useUpdatePrompt();

  const [selectedVersion, setSelectedVersion] = useState<string | null>(null);
  const selectedDetail = usePromptVersion(slug, selectedVersion);

  const [draft, setDraft] = useState("");
  const [dirty, setDirty] = useState(false);

  const isLocal = prompt.data?.registry_is_local ?? false;

  // When the prompt first loads, select its latest version and seed the draft.
  useEffect(() => {
    if (prompt.data && selectedVersion == null) {
      setSelectedVersion(String(prompt.data.latest_version));
      setDraft(prompt.data.template);
      setDirty(false);
    }
  }, [prompt.data, selectedVersion]);

  // When the user picks a non-latest version, swap the editor contents.
  useEffect(() => {
    if (selectedDetail.data) {
      setDraft(selectedDetail.data.template);
      setDirty(false);
    }
  }, [selectedDetail.data]);

  const variables = useMemo(() => extractVars(draft), [draft]);

  const handleSave = async () => {
    if (!slug || !draft.trim()) return;
    try {
      const res = await updatePrompt.mutateAsync({ slug, template: draft });
      toast.success(`Saved as v${res.version}`);
      setDirty(false);
      setSelectedVersion(String(res.version));
      queryClient.invalidateQueries({ queryKey: ["prompt", slug] });
      queryClient.invalidateQueries({ queryKey: ["prompt-versions", slug] });
      queryClient.invalidateQueries({ queryKey: ["prompts"] });
    } catch (e) {
      if (e instanceof ApiError) toast.error(e.message);
      else toast.error("Save failed");
    }
  };

  if (prompt.isLoading) return <TableSkeleton rows={4} />;
  if (prompt.error || !prompt.data) {
    return (
      <EmptyState
        title="Prompt not found"
        description="Check the slug or register it via code."
      />
    );
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title={prompt.data.slug}
        description={`Latest v${prompt.data.latest_version} · ${variables.length} variable${variables.length === 1 ? "" : "s"}`}
      >
        <Link to="/prompts">
          <Button variant="ghost" size="sm">
            <ChevronLeft className="mr-1.5 h-3.5 w-3.5" />
            Back
          </Button>
        </Link>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            prompt.refetch();
            versions.refetch();
            lineage.refetch();
          }}
          disabled={prompt.isFetching || versions.isFetching}
        >
          <RefreshCw
            className={`mr-1.5 h-3.5 w-3.5 ${
              prompt.isFetching || versions.isFetching ? "animate-spin" : ""
            }`}
          />
          Refresh
        </Button>
        <Button
          size="sm"
          onClick={handleSave}
          disabled={
            !isLocal || !dirty || updatePrompt.isPending || !draft.trim()
          }
          title={
            !isLocal
              ? "Registry is external — editing disabled"
              : !dirty
              ? "No changes to save"
              : undefined
          }
        >
          {updatePrompt.isPending ? (
            <>
              <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
              Saving…
            </>
          ) : (
            <>
              <Save className="mr-1.5 h-3.5 w-3.5" />
              Save as new version
            </>
          )}
        </Button>
      </PageHeader>

      {!isLocal && <RegistryExternalBanner />}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[220px_1fr]">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Versions</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            {versions.isLoading ? (
              <div className="p-3">
                <TableSkeleton rows={4} />
              </div>
            ) : (
              <ul className="divide-y">
                {(versions.data?.versions ?? []).map((v) => {
                  const isActive = selectedVersion === v.version;
                  return (
                    <li key={v.version}>
                      <button
                        type="button"
                        onClick={() => setSelectedVersion(v.version)}
                        className={cn(
                          "flex w-full flex-col items-start gap-0.5 px-3 py-2 text-left text-sm transition-colors",
                          isActive
                            ? "bg-primary/10 text-primary"
                            : "hover:bg-muted/50"
                        )}
                      >
                        <span className="font-mono">v{v.version}</span>
                        <span className="text-xs text-muted-foreground">
                          {formatTimeAgo(v.created_at)} · {v.created_by ?? "unknown"}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm flex items-center justify-between">
                <span>
                  Template
                  {selectedVersion && (
                    <span className="ml-2 font-mono text-xs text-muted-foreground">
                      (editing v{selectedVersion})
                    </span>
                  )}
                </span>
                {variables.length > 0 && (
                  <span className="font-mono text-xs font-normal text-muted-foreground">
                    vars: {variables.map((v) => `{{${v}}}`).join(" ")}
                  </span>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Textarea
                value={draft}
                onChange={(e) => {
                  setDraft(e.target.value);
                  setDirty(true);
                }}
                rows={16}
                className="font-mono text-sm"
                readOnly={!isLocal}
                placeholder="Enter prompt template. Use {{name}} for variables."
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Lineage</CardTitle>
            </CardHeader>
            <CardContent>
              {lineage.isLoading ? (
                <TableSkeleton rows={2} />
              ) : (
                <div className="space-y-3 text-sm">
                  <div>
                    <div className="mb-1 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
                      Traces using this prompt
                    </div>
                    {lineage.data?.trace_ids.length ? (
                      <ul className="flex flex-wrap gap-2">
                        {lineage.data.trace_ids.slice(0, 10).map((id) => (
                          <li key={id}>
                            <Link
                              to={`/traces/${id}`}
                              className="inline-flex items-center gap-1 rounded-md border px-2 py-1 font-mono text-xs hover:border-primary hover:text-primary"
                            >
                              {id.slice(0, 10)}…
                              <ExternalLink className="h-3 w-3" />
                            </Link>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-xs text-muted-foreground">None yet.</p>
                    )}
                  </div>
                  <div>
                    <div className="mb-1 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
                      Eval runs using this prompt
                    </div>
                    {lineage.data?.eval_run_ids.length ? (
                      <ul className="flex flex-wrap gap-2">
                        {lineage.data.eval_run_ids.slice(0, 10).map((id) => (
                          <li key={id}>
                            <Link
                              to={`/evals/${id}`}
                              className="inline-flex items-center gap-1 rounded-md border px-2 py-1 font-mono text-xs hover:border-primary hover:text-primary"
                            >
                              {id.slice(0, 10)}…
                              <ExternalLink className="h-3 w-3" />
                            </Link>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-xs text-muted-foreground">None yet.</p>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
