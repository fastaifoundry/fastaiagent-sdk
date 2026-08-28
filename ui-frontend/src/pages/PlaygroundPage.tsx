import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  ChevronDown,
  ChevronRight,
  ExternalLink,
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
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { ParametersPanel } from "@/components/playground/ParametersPanel";
import {
  RunHistory,
  type PlaygroundHistoryEntry,
} from "@/components/playground/RunHistory";
import { SaveAsEvalDialog } from "@/components/playground/SaveAsEvalDialog";
import { VariableForm } from "@/components/playground/VariableForm";
import {
  streamPlayground,
  usePlaygroundModels,
} from "@/hooks/use-playground";
import { usePrompt, usePrompts, usePromptVersion } from "@/hooks/use-prompts";
import { ApiError } from "@/lib/api";
import { formatCost, formatDurationMs, formatTokens } from "@/lib/format";
import type {
  PlaygroundDoneMetadata,
  PlaygroundParameters,
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

const VAR_RE = /\{\{(\w+)\}\}/g;

function extractVars(template: string): string[] {
  const matches = template.matchAll(VAR_RE);
  return Array.from(new Set(Array.from(matches, (m) => m[1])));
}

function resolveTemplate(
  template: string,
  values: Record<string, string>,
): string {
  let out = template;
  for (const [key, value] of Object.entries(values)) {
    out = out.split(`{{${key}}}`).join(value);
  }
  return out;
}

function uniqueId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function PlaygroundPage() {
  // ─── URL params: pre-fill from "Test in Playground" button on PromptEditorPage
  const [search, setSearch] = useSearchParams();
  const initialPromptSlug = search.get("prompt");
  const initialPromptVersion = search.get("version");

  const prompts = usePrompts();
  const models = usePlaygroundModels();

  // ─── Selected prompt + version
  const [selectedSlug, setSelectedSlug] = useState<string | null>(
    initialPromptSlug,
  );
  const [selectedVersion, setSelectedVersion] = useState<string | null>(
    initialPromptVersion,
  );

  const promptDetail = usePrompt(selectedSlug ?? undefined);
  const versionDetail = usePromptVersion(
    selectedSlug ?? undefined,
    selectedVersion,
  );

  // ─── Editable state — seeded from selection but freely edited
  const [systemPrompt, setSystemPrompt] = useState<string>("");
  const [systemOpen, setSystemOpen] = useState<boolean>(false);
  const [template, setTemplate] = useState<string>("");
  const [variables, setVariables] = useState<Record<string, string>>({});
  const [previewOpen, setPreviewOpen] = useState<boolean>(true);

  // Seed the editor when a prompt is selected. Non-latest version load
  // updates the template too so the user sees what they picked.
  useEffect(() => {
    if (selectedVersion && versionDetail.data) {
      setTemplate(versionDetail.data.template);
    } else if (promptDetail.data && !selectedVersion) {
      setTemplate(promptDetail.data.template);
      setSelectedVersion(String(promptDetail.data.latest_version));
    }
  }, [versionDetail.data, promptDetail.data, selectedVersion]);

  // ─── Provider / model
  const [provider, setProvider] = useState<string>("openai");
  const [model, setModel] = useState<string>("gpt-4o-mini");
  const providers = models.data?.providers ?? [];
  const providerInfo: PlaygroundProviderInfo | undefined = providers.find(
    (p) => p.provider === provider,
  );

  // Switching provider must repoint the model too. Leaving a stale
  // `gpt-4o-mini` selected while provider flipped to `anthropic` sent a
  // mismatched pair the provider rejects with a 404.
  const handleProviderChange = (next: string) => {
    setProvider(next);
    const info = providers.find((p) => p.provider === next);
    setModel(info?.models[0] ?? "");
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

  // ─── Parameters
  const [params, setParams] = useState<PlaygroundParameters>(DEFAULT_PARAMS);
  const [paramsOpen, setParamsOpen] = useState<boolean>(false);

  // ─── Connection: endpoint + per-run token.
  // Component state only — deliberately never written to localStorage or sent
  // anywhere but the run request, so a reload clears the token.
  const [baseUrl, setBaseUrl] = useState<string>("");
  const [apiKey, setApiKey] = useState<string>("");
  const [connOpen, setConnOpen] = useState<boolean>(false);
  const hasToken = apiKey.trim().length > 0;

  // A token supplied here stands in for the server's env var, so a provider
  // with no configured key is still runnable.
  const credentialed = Boolean(providerInfo?.has_key) || hasToken;

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

  const handleImagePick = (file: File | null) => {
    if (!file) {
      setImageBase64(null);
      setImageMediaType(null);
      setImageName(null);
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
      setImageBase64(result.slice(comma + 1));
      setImageMediaType(file.type);
      setImageName(file.name);
    };
    reader.readAsDataURL(file);
  };

  // ─── Run state
  const detectedVars = useMemo(() => extractVars(template), [template]);
  const resolvedPrompt = useMemo(
    () => resolveTemplate(template, variables),
    [template, variables],
  );

  const [response, setResponse] = useState<string>("");
  const [running, setRunning] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [errorCorrelationId, setErrorCorrelationId] = useState<string | null>(
    null,
  );
  const [lastMetadata, setLastMetadata] =
    useState<PlaygroundDoneMetadata | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const [history, setHistory] = useState<PlaygroundHistoryEntry[]>([]);
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null);

  const handleRun = useCallback(async () => {
    if (!template.trim()) {
      toast.error("Template is empty");
      return;
    }
    if (!credentialed) {
      const env = providerInfo?.env_var ?? `${provider.toUpperCase()}_API_KEY`;
      setError(
        `No API key found for provider '${provider}'. Set ${env} in your ` +
          `environment and restart the UI, or paste a token for this run in ` +
          `the Connection panel.`,
      );
      setErrorCorrelationId(null);
      return;
    }
    if (!model.trim()) {
      setError("Pick or type a model first.");
      setErrorCorrelationId(null);
      return;
    }
    setError(null);
    setErrorCorrelationId(null);
    setResponse("");
    setLastMetadata(null);
    setActiveHistoryId(null);
    setRunning(true);

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
          parameters: params,
          image_b64: imageBase64 ?? undefined,
          image_media_type: imageMediaType ?? undefined,
          base_url: baseUrl.trim() || undefined,
          api_key: apiKey.trim() || undefined,
        },
        ac.signal,
      )) {
        if (ev.event === "token") {
          collected += ev.text;
          setResponse(collected);
        } else if (ev.event === "done") {
          metadata = ev.metadata;
          setLastMetadata(ev.metadata);
        } else if (ev.event === "error") {
          setError(ev.message);
          setErrorCorrelationId(ev.correlation_id ?? null);
          break;
        }
      }
      if (metadata) {
        const entry: PlaygroundHistoryEntry = {
          id: uniqueId(),
          timestamp: new Date().toISOString(),
          model: metadata.model,
          provider: metadata.provider,
          response: collected,
          latency_ms: metadata.latency_ms,
          cost_usd: metadata.cost_usd,
          tokens: metadata.tokens,
          trace_id: metadata.trace_id,
          prompt_template: template,
          system_prompt: systemPrompt || null,
          variables,
        };
        setHistory((prev) => [entry, ...prev].slice(0, 25));
        setActiveHistoryId(entry.id);
      }
    } catch (e) {
      if ((e as { name?: string }).name === "AbortError") {
        // User clicked Stop — keep whatever has streamed so far.
      } else if (e instanceof ApiError) {
        setError(e.message);
      } else {
        setError(`Stream failed: ${(e as Error).message}`);
      }
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  }, [
    template,
    provider,
    model,
    variables,
    systemPrompt,
    params,
    imageBase64,
    imageMediaType,
    providerInfo,
    credentialed,
    baseUrl,
    apiKey,
  ]);

  // A model that isn't in the suggestion list is fine (the combobox accepts
  // free text and LLMClient takes any id the upstream API knows), but an
  // *empty* one never is.
  const canRun =
    Boolean(template.trim()) && Boolean(model.trim()) && credentialed;

  const handleStop = () => {
    abortRef.current?.abort();
  };

  const handleHistorySelect = (entry: PlaygroundHistoryEntry) => {
    setTemplate(entry.prompt_template);
    setSystemPrompt(entry.system_prompt ?? "");
    setVariables(entry.variables);
    setResponse(entry.response);
    setActiveHistoryId(entry.id);
    setLastMetadata({
      model: entry.model,
      provider: entry.provider,
      latency_ms: entry.latency_ms,
      tokens: entry.tokens,
      cost_usd: entry.cost_usd,
      trace_id: entry.trace_id,
    });
  };

  // Strip URL params after they've been consumed so a refresh stays put.
  useEffect(() => {
    if (initialPromptSlug || initialPromptVersion) {
      const next = new URLSearchParams(search);
      next.delete("prompt");
      next.delete("version");
      setSearch(next, { replace: true });
    }
    // Run once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-[1fr_140px]">
                <div className="space-y-1">
                  <Label htmlFor="playground-prompt" className="text-xs">
                    Prompt
                  </Label>
                  <Select
                    value={selectedSlug ?? ""}
                    onValueChange={(v) => {
                      setSelectedSlug(v);
                      setSelectedVersion(null);
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
                </div>
                <div className="space-y-1">
                  <Label htmlFor="playground-version" className="text-xs">
                    Version
                  </Label>
                  <Select
                    value={selectedVersion ?? ""}
                    onValueChange={setSelectedVersion}
                    disabled={!selectedSlug || !promptDetail.data}
                  >
                    <SelectTrigger id="playground-version" className="w-full">
                      <SelectValue placeholder="latest" />
                    </SelectTrigger>
                    <SelectContent>
                      {Array.from(
                        { length: promptDetail.data?.latest_version ?? 0 },
                        (_, i) => String(i + 1),
                      ).map((v) => (
                        <SelectItem key={v} value={v}>
                          v{v}
                        </SelectItem>
                      ))}
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
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-[140px_1fr]">
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
                              : undefined
                          }
                        >
                          {p.provider}
                          {!p.has_key && (
                            <span className="ml-1 text-[10px] text-muted-foreground">
                              (no key)
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
              </button>
              {paramsOpen && (
                <ParametersPanel value={params} onChange={setParams} />
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
                ) : (
                  "(default)"
                )}
              </button>
              {connOpen && (
                <div className="space-y-3 rounded-md border bg-muted/20 p-3">
                  <p className="text-xs text-muted-foreground">
                    For models behind an AI gateway or proxy. Both apply to
                    this run only — the token is never saved, logged, or sent
                    anywhere but the call itself, and a page reload clears it.
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
                    />
                    <p className="text-[11px] text-muted-foreground">
                      Leave empty to use {provider}&apos;s default.
                    </p>
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

          <div className="flex items-center justify-end gap-2">
            {running ? (
              <Button onClick={handleStop} variant="destructive">
                <Square className="mr-1.5 h-3.5 w-3.5" />
                Stop
              </Button>
            ) : (
              <Button onClick={handleRun} disabled={!canRun}>
                <Play className="mr-1.5 h-3.5 w-3.5" />
                Run
              </Button>
            )}
          </div>
        </div>

        {/* ─── RIGHT: Response ────────────────────────────────────────── */}
        <div className="space-y-4">
          <Card className={cn(error && "border-destructive")}>
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
              {error ? (
                <div className="space-y-1.5">
                  <pre className="rounded-md border border-destructive bg-destructive/5 p-3 font-mono text-xs whitespace-pre-wrap text-destructive">
                    {error}
                  </pre>
                  {errorCorrelationId && (
                    <p className="text-xs text-muted-foreground">
                      The provider&apos;s error is redacted here to avoid
                      leaking account details. The full text is in the server
                      log under correlation id{" "}
                      <code className="font-mono">{errorCorrelationId}</code>.
                    </p>
                  )}
                </div>
              ) : (
                <pre className="min-h-[160px] max-h-[480px] overflow-auto rounded-md border bg-muted/20 p-3 font-mono text-xs whitespace-pre-wrap">
                  {response || (
                    <span className="text-muted-foreground">
                      Press Run to call the LLM.
                    </span>
                  )}
                </pre>
              )}

              {lastMetadata && (
                <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                  <span className="font-mono">
                    {lastMetadata.provider}/{lastMetadata.model}
                  </span>
                  <span>{formatDurationMs(lastMetadata.latency_ms)}</span>
                  <span className="tabular-nums">
                    in {formatTokens(lastMetadata.tokens.input)} ·
                    out {formatTokens(lastMetadata.tokens.output)}
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
                    ~{formatCost(lastMetadata.cost_usd)} est.
                  </span>
                  {lastMetadata.trace_id && (
                    <Link
                      to={`/traces/${lastMetadata.trace_id}`}
                      className="inline-flex items-center gap-1 text-primary hover:underline"
                    >
                      <ExternalLink className="h-3 w-3" />
                      trace
                    </Link>
                  )}
                </div>
              )}

              <div className="flex justify-end">
                <SaveAsEvalDialog
                  resolvedInput={resolvedPrompt}
                  actualOutput={response}
                  systemPrompt={systemPrompt || null}
                  model={lastMetadata?.model ?? model}
                  provider={lastMetadata?.provider ?? provider}
                  disabled={!response || running}
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
