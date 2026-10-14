import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  ImageIcon,
  Loader2,
  Play,
  Square,
  X,
} from "lucide-react";
import { toast } from "sonner";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { ParametersPanel } from "@/components/playground/ParametersPanel";
import { PlaygroundTraceLink } from "@/components/playground/PlaygroundTraceLink";
import { isValidMaxTokens } from "@/lib/playground-params";
import { RunHistory } from "@/components/playground/RunHistory";
import { SaveAsEvalDialog } from "@/components/playground/SaveAsEvalDialog";
import { VariableForm } from "@/components/playground/VariableForm";
import {
  streamPlayground,
  usePlaygroundModels,
} from "@/hooks/use-playground";
import {
  usePrompt,
  usePrompts,
  usePromptVersion,
  usePromptVersions,
} from "@/hooks/use-prompts";
import { ApiError } from "@/lib/api";
import type { ErrorInfo } from "@/lib/api-error";
import { RunErrorPanel } from "@/components/playground/RunErrorPanel";
import { formatCost, formatDurationMs, formatTokens } from "@/lib/format";
import {
  extractVars,
  resolveTemplate,
  sameRequest,
  snapshotImage,
  unfilledVariables,
  type PlaygroundHistoryEntry,
  type PlaygroundRunSnapshot,
} from "@/lib/playground-run";
import {
  applySamplingRule,
  resolveSamplingRule,
  ruleLimits,
} from "@/lib/sampling-rules";
import type {
  PlaygroundDoneMetadata,
  PlaygroundParameters,
  PlaygroundPromptRef,
  PlaygroundProviderInfo,
} from "@/lib/types";
import { cn } from "@/lib/utils";

// temperature/top_p start unset on purpose — see PlaygroundParameters in
// lib/types.ts. Sending 1.0 defaults made every Anthropic call 400.
const DEFAULT_PARAMS: PlaygroundParameters = {
  temperature: null,
  max_tokens: 1024,
  top_p: null,
};

/** The version picker's value for an alias; a plain number is a version. */
const ALIAS_PREFIX = "alias:";

function uniqueId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

/** What the server accepts as an endpoint: an http(s) URL with a host. */
const ENDPOINT_RE = /^https?:\/\/[^/\s]+/;

/** A plain-message error, for the checks the page makes before any request. */
function messageError(message: string): ErrorInfo {
  return { message, fieldErrors: [] };
}

const varList = (names: string[]) => names.map((n) => `{{${n}}}`).join(", ");

export function PlaygroundPage() {
  // ─── URL params: pre-fill from "Test in Playground" on PromptEditorPage.
  // `?alias=production` picks the version the alias points at (1.89.0).
  const [search, setSearch] = useSearchParams();
  const initialPromptSlug = search.get("prompt");
  const initialPromptVersion = search.get("version");
  const initialPromptAlias = search.get("alias");

  const prompts = usePrompts();
  const models = usePlaygroundModels();

  // ─── Selected prompt + version (or alias)
  const [selectedSlug, setSelectedSlug] = useState<string | null>(
    initialPromptSlug,
  );
  const [versionChoice, setVersionChoice] = useState<string | null>(
    initialPromptAlias
      ? `${ALIAS_PREFIX}${initialPromptAlias}`
      : initialPromptVersion,
  );
  // Seed the template once the picked version loads. Only an explicit pick
  // seeds: a reactive seed overwrote a template restored from history.
  const seedPending = useRef<boolean>(Boolean(initialPromptSlug));

  const promptDetail = usePrompt(selectedSlug ?? undefined);
  const promptVersions = usePromptVersions(selectedSlug ?? undefined);
  const aliases = useMemo(
    () => promptDetail.data?.aliases ?? {},
    [promptDetail.data],
  );
  const choiceAlias = versionChoice?.startsWith(ALIAS_PREFIX)
    ? versionChoice.slice(ALIAS_PREFIX.length)
    : null;
  const choiceVersion: number | null = choiceAlias
    ? (aliases[choiceAlias] ?? null)
    : versionChoice
      ? Number(versionChoice)
      : null;
  const versionDetail = usePromptVersion(
    selectedSlug ?? undefined,
    choiceVersion != null ? String(choiceVersion) : null,
  );

  // ─── Editable state — seeded from selection but freely edited
  const [systemPrompt, setSystemPrompt] = useState<string>("");
  const [systemOpen, setSystemOpen] = useState<boolean>(false);
  const [template, setTemplate] = useState<string>("");
  const [variables, setVariables] = useState<Record<string, string>>({});
  const [previewOpen, setPreviewOpen] = useState<boolean>(true);

  useEffect(() => {
    if (!selectedSlug || !promptDetail.data) return;
    // No version picked, or an alias the prompt doesn't have: the latest.
    if (versionChoice == null || (choiceAlias && !(choiceAlias in aliases))) {
      setVersionChoice(String(promptDetail.data.latest_version));
      return;
    }
    if (
      seedPending.current &&
      versionDetail.data &&
      versionDetail.data.version === choiceVersion
    ) {
      setTemplate(versionDetail.data.template);
      seedPending.current = false;
    }
  }, [
    selectedSlug,
    promptDetail.data,
    versionDetail.data,
    versionChoice,
    choiceAlias,
    choiceVersion,
    aliases,
  ]);

  // Sent with the run so the server can attribute it to the prompt (P10c).
  const promptRef: PlaygroundPromptRef | null = selectedSlug
    ? {
        slug: selectedSlug,
        ...(choiceAlias
          ? { alias: choiceAlias }
          : choiceVersion != null
            ? { version: choiceVersion }
            : {}),
      }
    : null;

  // ─── Provider / model
  const [provider, setProvider] = useState<string>("openai");
  const [model, setModel] = useState<string>("gpt-4o-mini");
  const providers = models.data?.providers ?? [];
  const providerInfo: PlaygroundProviderInfo | undefined = providers.find(
    (p) => p.provider === provider,
  );

  // ─── Connection: endpoint + per-run token.
  // Component state only — deliberately never written to localStorage or sent
  // anywhere but the run request, so a reload clears the token.
  const [baseUrl, setBaseUrl] = useState<string>("");
  const [apiKey, setApiKey] = useState<string>("");
  const [connOpen, setConnOpen] = useState<boolean>(false);
  const hasToken = apiKey.trim().length > 0;

  // Switching provider must repoint the model too. Leaving a stale
  // `gpt-4o-mini` selected while provider flipped to `anthropic` sent a
  // mismatched pair the provider rejects with a 404.
  const handleProviderChange = (next: string) => {
    setProvider(next);
    const info = providers.find((p) => p.provider === next);
    setModel(info?.models[0] ?? "");
    // custom/azure have no default endpoint: open where it is set (P10i).
    if (info?.requires_endpoint) setConnOpen(true);
  };

  // First time we get models, default to a provider that has a key.
  useEffect(() => {
    if (!models.data) return;
    const withKey = models.data.providers.find((p) => p.has_key);
    if (withKey && withKey.provider !== provider) {
      // Only auto-switch on first load; respect user choice afterwards.
      const picked = providers.find((p) => p.provider === provider);
      if (!picked || !picked.has_key) {
        setProvider(withKey.provider);
        setModel(withKey.models[0] ?? model);
      }
    }
  }, [models.data]);  // eslint-disable-line react-hooks/exhaustive-deps

  // ─── Parameters, and what the model accepts (P7: hints, never a block)
  const [params, setParams] = useState<PlaygroundParameters>(DEFAULT_PARAMS);
  const [paramsOpen, setParamsOpen] = useState<boolean>(false);
  const rule = useMemo(
    () => resolveSamplingRule(models.data?.sampling_rules, provider, model),
    [models.data, provider, model],
  );
  const sentParams = useMemo(() => applySamplingRule(params, rule), [params, rule]);

  // A token supplied here stands in for the server's env var, so a provider
  // with no configured key is still runnable.
  const credentialed = Boolean(providerInfo?.has_key) || hasToken;
  const endpointMissing =
    Boolean(providerInfo?.requires_endpoint) && !baseUrl.trim();

  // Anything other than a loopback origin means the token would cross the
  // network, and the local UI serves plain HTTP.
  const insecureOrigin =
    typeof window !== "undefined" &&
    !["localhost", "127.0.0.1", "[::1]", "::1"].includes(
      window.location.hostname,
    );

  // ─── Image attachment (multimodal)
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [imageMediaType, setImageMediaType] = useState<string | null>(null);
  const [imageName, setImageName] = useState<string | null>(null);

  const setImage = (b64: string | null, type: string | null, name: string | null) => {
    setImageBase64(b64);
    setImageMediaType(type);
    setImageName(name);
  };

  const handleImagePick = (file: File | null) => {
    if (!file) {
      setImage(null, null, null);
      return;
    }
    if (
      !["image/jpeg", "image/png", "image/gif", "image/webp"].includes(file.type)
    ) {
      toast.error("Only JPEG/PNG/GIF/WebP supported");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // result is "data:<mime>;base64,<payload>" — strip the prefix.
      const comma = result.indexOf(",");
      setImage(result.slice(comma + 1), file.type, file.name);
    };
    reader.readAsDataURL(file);
  };

  // ─── Run state
  const detectedVars = useMemo(() => extractVars(template), [template]);
  const resolvedPrompt = useMemo(
    () => resolveTemplate(template, variables),
    [template, variables],
  );
  const unfilledNow = useMemo(
    () => unfilledVariables(template, variables),
    [template, variables],
  );

  // The request the form would send now, in the shape a run keeps (P6).
  const formSnapshot: PlaygroundRunSnapshot = {
    provider,
    model,
    prompt_template: template,
    variables,
    system_prompt: systemPrompt || null,
    parameters: sentParams,
    base_url: baseUrl.trim() || null,
    prompt_ref: promptRef,
    resolved_prompt: resolvedPrompt,
    image: snapshotImage(imageBase64, imageMediaType, imageName),
  };

  const [response, setResponse] = useState<string>("");
  const [running, setRunning] = useState<boolean>(false);
  // One normalised error (1.89.0, P1). It used to be the raw `detail`, which is
  // a list for a 422 — rendering it crashed the page and lost the form.
  const [errorInfo, setErrorInfo] = useState<ErrorInfo | null>(null);
  // Set when Run was pressed with unfilled {{vars}}: warn, don't block (D10).
  const [unfilledWarning, setUnfilledWarning] = useState<string[] | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const [history, setHistory] = useState<PlaygroundHistoryEntry[]>([]);
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null);
  // Everything under the response reads the run, never the form.
  const activeRun = history.find((h) => h.id === activeHistoryId) ?? null;
  const shownMetadata: PlaygroundDoneMetadata | null = activeRun?.metadata ?? null;
  const formChanged = activeRun ? !sameRequest(activeRun.request, formSnapshot) : false;

  const handleRun = async (force = false) => {
    if (!template.trim()) {
      toast.error("Template is empty");
      return;
    }
    if (!credentialed) {
      const env = providerInfo?.env_var ?? `${provider.toUpperCase()}_API_KEY`;
      setErrorInfo(
        messageError(
          `No API key found for provider '${provider}'. Set ${env} in your ` +
            `environment and restart the UI, or paste a token for this run in ` +
            `the Connection panel.`,
        ),
      );
      return;
    }
    if (!model.trim()) {
      setErrorInfo(messageError("Pick or type a model first."));
      return;
    }
    if (!force && unfilledNow.length > 0) {
      setUnfilledWarning(unfilledNow);
      return;
    }
    setUnfilledWarning(null);
    setErrorInfo(null);
    setResponse("");
    setActiveHistoryId(null);
    setRunning(true);

    // Captured before streaming: the history entry, the Response card and
    // "Save as eval case" all read this, whatever the form becomes.
    const snapshot: PlaygroundRunSnapshot = {
      ...formSnapshot,
      variables: { ...variables },
    };

    const ac = new AbortController();
    abortRef.current = ac;

    let collected = "";
    let metadata: PlaygroundDoneMetadata | null = null;

    try {
      for await (const ev of streamPlayground(
        {
          provider,
          model,
          prompt_template: template,
          variables,
          system_prompt: systemPrompt || undefined,
          parameters: sentParams,
          image_b64: imageBase64 ?? undefined,
          image_media_type: imageMediaType ?? undefined,
          base_url: baseUrl.trim() || undefined,
          api_key: apiKey.trim() || undefined,
          prompt_ref: promptRef ?? undefined,
        },
        ac.signal,
      )) {
        if (ev.event === "token") {
          collected += ev.text;
          setResponse(collected);
        } else if (ev.event === "done") {
          metadata = ev.metadata;
        } else if (ev.event === "error") {
          setErrorInfo({
            message: ev.message,
            fieldErrors: ev.field_errors ?? [],
            correlationId: ev.correlation_id,
            kind: ev.kind,
            providerStatus: ev.provider_status,
            providerMessage: ev.provider_message,
            hint: ev.hint,
          });
          break;
        }
      }
      if (metadata) {
        const entry: PlaygroundHistoryEntry = {
          id: uniqueId(),
          timestamp: new Date().toISOString(),
          request: snapshot,
          response: collected,
          metadata,
        };
        setHistory((prev) => [entry, ...prev].slice(0, 25));
        setActiveHistoryId(entry.id);
      }
    } catch (e) {
      if ((e as { name?: string }).name === "AbortError") {
        // User clicked Stop — keep whatever has streamed so far.
      } else if (e instanceof ApiError) {
        setErrorInfo(e.info);
      } else {
        setErrorInfo(messageError(`Stream failed: ${(e as Error).message}`));
      }
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  };

  // A model that isn't in the suggestion list is fine (the combobox accepts
  // free text and LLMClient takes any id the upstream API knows), but an
  // *empty* one never is.
  // A bad max_tokens or endpoint is caught here rather than sent: the server's
  // 422 for it used to crash the page (1.89.0, P1).
  const endpointValid = !baseUrl.trim() || ENDPOINT_RE.test(baseUrl.trim());
  const canRun =
    Boolean(template.trim()) &&
    Boolean(model.trim()) &&
    credentialed &&
    isValidMaxTokens(params.max_tokens) &&
    endpointValid &&
    !endpointMissing;

  const handleStop = () => {
    abortRef.current?.abort();
  };

  // Restore the form exactly as the run sent it (P10d). Never the token.
  const handleHistorySelect = (entry: PlaygroundHistoryEntry) => {
    const r = entry.request;
    seedPending.current = false;
    setProvider(r.provider);
    setModel(r.model);
    setParams(r.parameters);
    setSystemPrompt(r.system_prompt ?? "");
    setBaseUrl(r.base_url ?? "");
    setTemplate(r.prompt_template);
    setVariables(r.variables);
    setSelectedSlug(r.prompt_ref?.slug ?? null);
    setVersionChoice(
      r.prompt_ref?.alias
        ? `${ALIAS_PREFIX}${r.prompt_ref.alias}`
        : r.prompt_ref?.version != null
          ? String(r.prompt_ref.version)
          : null,
    );
    if (r.image?.b64) {
      setImage(r.image.b64, r.image.media_type, r.image.name);
    } else {
      setImage(null, null, null);
      if (r.image) {
        toast.info(`${r.image.name} was over 5 MB and was not kept. Attach it again.`);
      }
    }
    setResponse(entry.response);
    setErrorInfo(null);
    setUnfilledWarning(null);
    setActiveHistoryId(entry.id);
  };

  // Strip URL params after they've been consumed so a refresh stays put.
  useEffect(() => {
    if (initialPromptSlug || initialPromptVersion || initialPromptAlias) {
      const next = new URLSearchParams(search);
      next.delete("prompt");
      next.delete("version");
      next.delete("alias");
      setSearch(next, { replace: true });
    }
    // Run once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const versionRows = promptVersions.data?.versions ?? [];
  const aliasNames = Object.keys(aliases).sort();

  return (
    <div className="space-y-5">
      <PageHeader
        title="Playground"
        description="Pick a prompt, fill the variables, run an LLM call. History clears on refresh."
      />

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {/* ─── LEFT: Configuration ────────────────────────────────────── */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">// PROMPT</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-[1fr_170px]">
                <div className="space-y-1">
                  <Label htmlFor="playground-prompt" className="text-xs">
                    Prompt
                  </Label>
                  <div className="flex items-center gap-1">
                    <Select
                      value={selectedSlug ?? ""}
                      onValueChange={(v) => {
                        seedPending.current = true;
                        setSelectedSlug(v);
                        setVersionChoice(null);
                        setVariables({});
                      }}
                    >
                      <SelectTrigger id="playground-prompt" className="w-full">
                        <SelectValue placeholder="Select a prompt…" />
                      </SelectTrigger>
                      <SelectContent>
                        {(prompts.data?.rows ?? []).map((p) => (
                          <SelectItem key={p.name} value={p.name}>
                            {p.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    {selectedSlug && (
                      <button
                        type="button"
                        onClick={() => {
                          setSelectedSlug(null);
                          setVersionChoice(null);
                        }}
                        title="Detach from the registry prompt (the template stays)"
                        aria-label="Detach prompt"
                        className="text-muted-foreground hover:text-foreground"
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    )}
                  </div>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="playground-version" className="text-xs">
                    Version
                  </Label>
                  {/* Real versions (they need not be 1..latest) plus aliases. */}
                  <Select
                    value={versionChoice ?? ""}
                    onValueChange={(v) => {
                      seedPending.current = true;
                      setVersionChoice(v);
                    }}
                    disabled={!selectedSlug || !promptDetail.data}
                  >
                    <SelectTrigger id="playground-version" className="w-full">
                      <SelectValue placeholder="latest" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        <SelectLabel>Versions</SelectLabel>
                        {versionRows.map((v) => (
                          <SelectItem key={v.version} value={String(v.version)}>
                            v{v.version}
                          </SelectItem>
                        ))}
                      </SelectGroup>
                      {aliasNames.length > 0 && (
                        <SelectGroup>
                          <SelectLabel>Aliases</SelectLabel>
                          {aliasNames.map((a) => (
                            <SelectItem key={a} value={`${ALIAS_PREFIX}${a}`}>
                              {a} → v{aliases[a]}
                            </SelectItem>
                          ))}
                        </SelectGroup>
                      )}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <button
                type="button"
                className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                onClick={() => setSystemOpen((v) => !v)}
              >
                {systemOpen ? (
                  <ChevronDown className="h-3 w-3" />
                ) : (
                  <ChevronRight className="h-3 w-3" />
                )}
                System prompt {systemPrompt ? "(set)" : "(none)"}
              </button>
              {systemOpen && (
                <Textarea
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  placeholder="You are a helpful assistant…"
                  rows={3}
                  className="font-mono text-xs"
                />
              )}

              <div className="space-y-1">
                <Label htmlFor="playground-template" className="text-xs">
                  Template
                </Label>
                <Textarea
                  id="playground-template"
                  value={template}
                  onChange={(e) => setTemplate(e.target.value)}
                  rows={8}
                  className="font-mono text-xs"
                  placeholder={"Hi {{name}}, how can I help with {{topic}}?"}
                />
              </div>

              <div className="space-y-1">
                <Label className="text-xs">Variables</Label>
                <VariableForm
                  variables={detectedVars}
                  values={variables}
                  onChange={setVariables}
                  unfilled={unfilledNow}
                />
              </div>

              <button
                type="button"
                className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                onClick={() => setPreviewOpen((v) => !v)}
              >
                {previewOpen ? (
                  <ChevronDown className="h-3 w-3" />
                ) : (
                  <ChevronRight className="h-3 w-3" />
                )}
                Resolved preview (this is what the LLM sees)
              </button>
              {previewOpen && (
                <pre className="max-h-64 overflow-auto rounded-md border bg-muted/30 p-2 font-mono text-xs whitespace-pre-wrap">
                  {resolvedPrompt || "(empty)"}
                </pre>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">// MODEL</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-[170px_1fr]">
                <div className="space-y-1">
                  <Label htmlFor="playground-provider" className="text-xs">
                    Provider
                  </Label>
                  <Select value={provider} onValueChange={handleProviderChange}>
                    <SelectTrigger id="playground-provider" className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {providers.map((p) => (
                        <SelectItem
                          key={p.provider}
                          value={p.provider}
                          disabled={!p.has_key}
                          title={
                            !p.has_key && p.env_var
                              ? `Set ${p.env_var} to enable`
                              : p.requires_endpoint
                                ? "No default endpoint: set one in the Connection panel"
                                : undefined
                          }
                        >
                          {p.provider}
                          {!p.has_key && (
                            <span className="ml-1 text-[10px] text-muted-foreground">
                              (no key)
                            </span>
                          )}
                          {p.has_key && p.requires_endpoint && (
                            <span className="ml-1 text-[10px] text-muted-foreground">
                              (needs endpoint)
                            </span>
                          )}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="playground-model" className="text-xs">
                    Model
                  </Label>
                  {/* Combobox, not a fixed list: model ids go stale between
                      releases, and LLMClient accepts any id the upstream API
                      knows. Suggestions come from models.json + the shipped
                      catalog; typing anything else is allowed. */}
                  <Input
                    id="playground-model"
                    list="playground-model-options"
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    placeholder="Pick or type a model id…"
                    autoComplete="off"
                    spellCheck={false}
                    className="font-mono text-xs"
                  />
                  <datalist id="playground-model-options">
                    {(providerInfo?.models ?? []).map((m) => (
                      <option key={m} value={m} />
                    ))}
                  </datalist>
                </div>
              </div>

              <button
                type="button"
                className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                onClick={() => setParamsOpen((v) => !v)}
              >
                {paramsOpen ? (
                  <ChevronDown className="h-3 w-3" />
                ) : (
                  <ChevronRight className="h-3 w-3" />
                )}
                Parameters
                {ruleLimits(rule) && (
                  <span className="text-amber-700 dark:text-amber-300">
                    (limited for this model)
                  </span>
                )}
              </button>
              {paramsOpen && (
                <ParametersPanel value={params} onChange={setParams} rule={rule} />
              )}

              <button
                type="button"
                className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                onClick={() => setConnOpen((v) => !v)}
              >
                {connOpen ? (
                  <ChevronDown className="h-3 w-3" />
                ) : (
                  <ChevronRight className="h-3 w-3" />
                )}
                Connection{" "}
                {baseUrl || hasToken ? (
                  <span className="text-primary">
                    ({[baseUrl && "endpoint", hasToken && "token"]
                      .filter(Boolean)
                      .join(" + ")}
                    )
                  </span>
                ) : endpointMissing ? (
                  <span className="text-destructive">(endpoint needed)</span>
                ) : (
                  "(default)"
                )}
              </button>
              {connOpen && (
                <div className="space-y-3 rounded-md border bg-muted/20 p-3">
                  <p className="text-xs text-muted-foreground">
                    For models behind an AI gateway or proxy. Both apply to
                    this run only. The token is never saved or kept with a
                    run, and a page reload clears it.
                  </p>
                  <div className="space-y-1">
                    <Label htmlFor="playground-base-url" className="text-xs">
                      Endpoint
                    </Label>
                    <Input
                      id="playground-base-url"
                      value={baseUrl}
                      onChange={(e) => setBaseUrl(e.target.value)}
                      placeholder="https://ai-gateway.internal/v1"
                      autoComplete="off"
                      spellCheck={false}
                      className="font-mono text-xs"
                      aria-invalid={!endpointValid || endpointMissing}
                    />
                    {!endpointValid ? (
                      <p className="text-[11px] text-destructive" role="alert">
                        The endpoint must be an http:// or https:// URL.
                      </p>
                    ) : endpointMissing ? (
                      <p className="text-[11px] text-destructive">
                        {provider} has no default endpoint. Enter its base URL.
                      </p>
                    ) : (
                      <p className="text-[11px] text-muted-foreground">
                        Leave empty to use {provider}&apos;s default.
                      </p>
                    )}
                  </div>
                  <div className="space-y-1">
                    <Label htmlFor="playground-api-key" className="text-xs">
                      Token
                    </Label>
                    <Input
                      id="playground-api-key"
                      type="password"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder={
                        providerInfo?.env_var
                          ? `Overrides ${providerInfo.env_var}`
                          : "Bearer token for this run"
                      }
                      autoComplete="off"
                      spellCheck={false}
                      className="font-mono text-xs"
                    />
                  </div>
                  {baseUrl.trim() && !hasToken && providerInfo?.has_key && (
                    <p className="rounded-md border border-amber-500/50 bg-amber-500/5 p-2 text-[11px]">
                      A custom endpoint needs its own token. Without one the
                      server would send{" "}
                      <code className="font-mono">
                        {providerInfo.env_var ?? "its provider key"}
                      </code>{" "}
                      to this endpoint, which is only right if the endpoint
                      really is {provider}. The run is refused until you enter
                      a token or clear the endpoint.
                    </p>
                  )}
                  {insecureOrigin && hasToken && (
                    <p className="rounded-md border border-destructive/50 bg-destructive/5 p-2 text-[11px] text-destructive">
                      This UI is not on localhost. The local UI serves plain
                      HTTP, so a token typed here crosses the network
                      unencrypted. Prefer an environment variable on the
                      server.
                    </p>
                  )}
                </div>
              )}

              <div className="flex items-center gap-2 pt-1">
                <label className="inline-flex cursor-pointer items-center gap-2 rounded-md border border-input bg-transparent px-3 py-1.5 text-xs hover:bg-muted/50">
                  <ImageIcon className="h-3.5 w-3.5" />
                  Attach image
                  <input
                    type="file"
                    accept="image/jpeg,image/png,image/gif,image/webp"
                    className="hidden"
                    onChange={(e) => handleImagePick(e.target.files?.[0] ?? null)}
                  />
                </label>
                {imageName && imageBase64 && imageMediaType && (
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <img
                      src={`data:${imageMediaType};base64,${imageBase64}`}
                      alt={imageName}
                      className="h-8 w-8 rounded object-cover"
                    />
                    <span className="font-mono">{imageName}</span>
                    <button
                      type="button"
                      onClick={() => handleImagePick(null)}
                      className="text-muted-foreground hover:text-foreground"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {unfilledWarning && (
            <div
              className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-amber-500/40 bg-amber-500/5 px-3 py-2 text-xs text-amber-900 dark:text-amber-200"
              role="alert"
              data-testid="unfilled-warning"
            >
              <span>
                Not filled: <span className="font-mono">{varList(unfilledWarning)}</span>.
                The model would see the placeholders as written.
              </span>
              <span className="flex gap-2">
                <Button size="sm" variant="ghost" onClick={() => setUnfilledWarning(null)}>
                  Cancel
                </Button>
                <Button size="sm" variant="outline" onClick={() => handleRun(true)}>
                  Run anyway
                </Button>
              </span>
            </div>
          )}

          <div className="flex items-center justify-end gap-2">
            {endpointMissing && (
              <span className="text-xs text-muted-foreground" data-testid="run-blocked-reason">
                {provider} needs an Endpoint (Connection panel).
              </span>
            )}
            {running ? (
              <Button onClick={handleStop} variant="destructive">
                <Square className="mr-1.5 h-3.5 w-3.5" />
                Stop
              </Button>
            ) : (
              <Button onClick={() => handleRun()} disabled={!canRun}>
                <Play className="mr-1.5 h-3.5 w-3.5" />
                Run
              </Button>
            )}
          </div>
        </div>

        {/* ─── RIGHT: Response ────────────────────────────────────────── */}
        <div className="space-y-4">
          <Card className={cn(errorInfo && "border-destructive")}>
            <CardHeader>
              <CardTitle className="flex items-center justify-between text-sm">
                <span>// RESPONSE</span>
                {running && (
                  <span className="flex items-center gap-1 text-xs font-normal text-muted-foreground">
                    <Loader2 className="h-3 w-3 animate-spin" />
                    streaming…
                  </span>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {shownMetadata?.truncated && (
                <p
                  className="flex items-center gap-1.5 rounded-md border border-amber-500/40 bg-amber-500/5 px-3 py-2 text-xs text-amber-900 dark:text-amber-200"
                  data-testid="truncated-banner"
                >
                  <AlertTriangle className="h-3.5 w-3.5" />
                  Cut off at max_tokens (
                  {shownMetadata.max_tokens ?? activeRun?.request.parameters.max_tokens}
                  ). Raise max_tokens to get the rest.
                </p>
              )}
              {shownMetadata?.unfilled_variables &&
                shownMetadata.unfilled_variables.length > 0 && (
                  <p
                    className="text-xs text-amber-700 dark:text-amber-300"
                    data-testid="sent-unfilled"
                  >
                    Sent with unfilled variables:{" "}
                    <span className="font-mono">
                      {varList(shownMetadata.unfilled_variables)}
                    </span>
                  </p>
                )}
              {errorInfo ? (
                <RunErrorPanel info={errorInfo} />
              ) : (
                <pre className="min-h-[160px] max-h-[480px] overflow-auto rounded-md border bg-muted/20 p-3 font-mono text-xs whitespace-pre-wrap">
                  {response || (
                    <span className="text-muted-foreground">
                      Press Run to call the LLM.
                    </span>
                  )}
                </pre>
              )}

              {shownMetadata && (
                <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                  <span className="font-mono">
                    {shownMetadata.provider}/{shownMetadata.model}
                  </span>
                  <span>{formatDurationMs(shownMetadata.latency_ms)}</span>
                  <span className="tabular-nums">
                    in {formatTokens(shownMetadata.tokens.input)} ·
                    out {formatTokens(shownMetadata.tokens.output)}
                  </span>
                  <span
                    title={
                      "Estimated from public list prices. Does not account for " +
                      "negotiated discounts, Bedrock/Vertex partner rates, the " +
                      "Batch API discount, or prompt-cache multipliers. Set your " +
                      "own rates in the 'pricing' block of .fastaiagent/models.json."
                    }
                    className="cursor-help border-b border-dotted border-muted-foreground/50"
                  >
                    ~{formatCost(shownMetadata.cost_usd)} est.
                  </span>
                  {shownMetadata.prompt && (
                    <span
                      data-testid="prompt-attribution"
                      title={
                        shownMetadata.prompt.attributed
                          ? "Sent as stored: the run is in this prompt's lineage."
                          : "The template was edited, so the run is not attributed to the version."
                      }
                    >
                      {shownMetadata.prompt.attributed ? "attributed to " : "modified from "}
                      <span className="font-mono">
                        {shownMetadata.prompt.slug}
                        {shownMetadata.prompt.version != null &&
                          ` v${shownMetadata.prompt.version}`}
                      </span>
                    </span>
                  )}
                  <PlaygroundTraceLink
                    traceId={shownMetadata.trace_id}
                    inUiDb={shownMetadata.trace_in_ui_db}
                  />
                </div>
              )}

              <div className="flex justify-end">
                <SaveAsEvalDialog
                  run={activeRun}
                  formChanged={formChanged}
                  disabled={running}
                />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">// HISTORY</CardTitle>
            </CardHeader>
            <CardContent>
              <RunHistory
                entries={history}
                activeId={activeHistoryId}
                onSelect={handleHistorySelect}
              />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
