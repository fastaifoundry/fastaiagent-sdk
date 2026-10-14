import { useState } from "react";
import { Loader2, Play } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { ApiError } from "@/lib/api";
import { cn } from "@/lib/utils";
import type {
  ReplayDeterminism,
  ReplayOnMiss,
  ReplayToolChoice,
  RerunResult,
  ReplayStep,
} from "@/lib/types";
import { useForkAtStep, useModifyFork, useRerunFork } from "@/hooks/use-replay";

interface Props {
  traceId: string;
  step: ReplayStep | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onRerunComplete: (fork_id: string, result: RerunResult) => void;
  /**
   * The original agent's tools (``GET /api/replay/{id}`` → ``tools``,
   * 1.89.0). Each gets a Block / Stub / Live choice; a rerun runs only the
   * tools you stub or allow.
   */
  tools?: ReplayToolChoice[];
}

type ToolMode = "block" | "stub" | "live";

const SELECT_CLASS =
  "h-9 w-full rounded-md border border-input bg-background px-2 text-sm";

/** Why Live is unavailable for this tool, or null when it can run live. */
function liveBlockedReason(tool: ReplayToolChoice): string | null {
  if (tool.credentials_redacted) {
    return "Its credentials were redacted from the trace, so a live call cannot authenticate. Stub it instead.";
  }
  if (!tool.live_available && !tool.endpoint_from_trace) {
    return "This UI process has no implementation of this tool. Start the UI with `fastaiagent ui --agent …` to run it live, or stub it.";
  }
  return null;
}

/** A stub response: JSON when it parses, else the raw text. */
function parseStub(value: string): unknown {
  const trimmed = value.trim();
  try {
    return JSON.parse(trimmed);
  } catch {
    return value;
  }
}

function tryParse(value: string): Record<string, unknown> | null {
  const trimmed = value.trim();
  if (!trimmed) return {};
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return null;
  } catch {
    return null;
  }
}

/** Parse a value that the user may have typed as either a JSON object or a
 * JSON list of multimodal parts. Returns the value to send as ``input`` on
 * the modify endpoint, or ``null`` if the JSON is malformed. */
function tryParseInput(value: string): unknown {
  const trimmed = value.trim();
  if (!trimmed) return {};
  try {
    return JSON.parse(trimmed);
  } catch {
    return null;
  }
}

/** Read a File via FileReader and return its base64 payload (no prefix). */
async function fileToBase64(file: File): Promise<string> {
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error);
    reader.onload = () => {
      const result = reader.result as string;
      // result looks like "data:image/jpeg;base64,...."
      const comma = result.indexOf(",");
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.readAsDataURL(file);
  });
}

export function ReplayForkDialog({
  traceId,
  step,
  open,
  onOpenChange,
  onRerunComplete,
  tools = [],
}: Props) {
  const [prompt, setPrompt] = useState("");
  // 1.89.0: how the rerun calls the LLM, and what it may execute.
  const [determinism, setDeterminism] = useState<ReplayDeterminism>("live");
  const [onMiss, setOnMiss] = useState<ReplayOnMiss>("live");
  // Per-tool choice; a tool absent from the map is blocked (the default).
  const [toolModes, setToolModes] = useState<Record<string, ToolMode>>({});
  const [stubTexts, setStubTexts] = useState<Record<string, string>>({});
  const [inputJson, setInputJson] = useState("");
  // v1.14.1+: ``toolName`` + ``toolJson`` together build a single entry
  // in ``mods.tool_overrides`` posted to the route. The previous shape
  // (just a JSON blob sent as ``tool_response``) was silently dropped
  // for agent reruns — see PR #80 / docs/replay/index.md.
  const [toolName, setToolName] = useState("");
  const [toolJson, setToolJson] = useState("");
  const [temperature, setTemperature] = useState<string>("");
  const [maxTokens, setMaxTokens] = useState<string>("");

  const fork = useForkAtStep(traceId);
  const modify = useModifyFork();
  const rerun = useRerunFork();

  const pending = fork.isPending || modify.isPending || rerun.isPending;

  const reset = () => {
    setPrompt("");
    setInputJson("");
    setToolName("");
    setToolJson("");
    setTemperature("");
    setMaxTokens("");
    setDeterminism("live");
    setOnMiss("live");
    setToolModes({});
    setStubTexts({});
  };

  const modeOf = (name: string): ToolMode => toolModes[name] ?? "block";
  const counts = tools.reduce(
    (acc, t) => {
      acc[modeOf(t.name)] += 1;
      return acc;
    },
    { block: 0, stub: 0, live: 0 } as Record<ToolMode, number>
  );

  const handleRerun = async () => {
    if (!step) return;

    // Validate JSON inputs up front so we fail before committing a fork.
    // ``input`` accepts either a JSON object (legacy form) or a list of
    // typed parts (multimodal — see ``ReplayForkDialog`` Image tab below).
    const parsedInput = inputJson ? tryParseInput(inputJson) : null;
    if (inputJson && parsedInput === null) {
      toast.error("Input must be valid JSON object or list of parts");
      return;
    }
    const parsedTool = toolJson ? tryParse(toolJson) : null;
    if (toolJson && parsedTool === null) {
      toast.error("Tool response must be valid JSON object");
      return;
    }
    if (toolJson && !toolName.trim()) {
      // v1.14.1+: backend needs a name to call with_tool_override(name, …)
      toast.error("Tool name is required when providing a tool response override");
      return;
    }
    // 1.89.0: Stub → tool_overrides (a canned response, this rerun only);
    // Live → allow_tools; everything else stays blocked.
    const stubs: Record<string, unknown> = {};
    const allowTools: string[] = [];
    for (const t of tools) {
      const mode = modeOf(t.name);
      if (mode === "stub") {
        const text = stubTexts[t.name] ?? "";
        if (!text.trim()) {
          toast.error(`Enter a stub response for ${t.name}, or block it`);
          return;
        }
        stubs[t.name] = parseStub(text);
      } else if (mode === "live" && liveBlockedReason(t) === null) {
        allowTools.push(t.name);
      }
    }

    try {
      const { fork_id } = await fork.mutateAsync(step.step);

      const mods: Record<string, unknown> = {};
      if (prompt) mods.prompt = prompt;
      if (parsedInput !== null) {
        if (Array.isArray(parsedInput)) {
          // List of multimodal parts — send as-is; the backend resolver
          // turns each ``{"type":"image","data_base64":...}`` entry into
          // a real ``Image`` before the rerun.
          if (parsedInput.length > 0) mods.input = parsedInput;
        } else if (
          parsedInput &&
          typeof parsedInput === "object" &&
          Object.keys(parsedInput as Record<string, unknown>).length > 0
        ) {
          mods.input = parsedInput;
        }
      }
      // v1.14.1: send as tool_overrides so the route wires it through
      // ForkedReplay.with_tool_override (the v1.13 tool_response field
      // was silently dropped for agent reruns — backend keeps accepting
      // it with a deprecation warning).
      const overrides: Record<string, unknown> = { ...stubs };
      if (parsedTool && toolName.trim()) {
        overrides[toolName.trim()] = parsedTool;
      }
      if (Object.keys(overrides).length > 0) mods.tool_overrides = overrides;

      const config: Record<string, unknown> = {};
      if (temperature !== "") config.temperature = Number(temperature);
      if (maxTokens !== "") config.max_tokens = Number(maxTokens);
      if (Object.keys(config).length > 0) mods.config = config;

      // Always sent, so the rerun does what this dialog shows: unchosen
      // tools are blocked, whatever the server's default policy is.
      mods.determinism = determinism;
      if (determinism === "recorded") mods.on_miss = onMiss;
      mods.tool_policy = "block";
      mods.allow_tools = allowTools;

      await modify.mutateAsync({ forkId: fork_id, mods });

      const result = await rerun.mutateAsync(fork_id);
      toast.success("Rerun complete");
      onRerunComplete(fork_id, result);
      reset();
      onOpenChange(false);
    } catch (e) {
      if (e instanceof ApiError) toast.error(e.message);
      else toast.error("Fork-and-rerun failed");
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next) reset();
        onOpenChange(next);
      }}
    >
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Fork and rerun with modifications</DialogTitle>
          <DialogDescription>
            {step ? (
              <>
                Re-runs the whole agent from the top with your changes
                applied — see step <span className="font-mono">{step.step}</span>{" "}
                <span className="font-mono text-foreground">{step.span_name}</span>{" "}
                as the divergence reference. True mid-trace resume is a
                v2 concern; the agent re-executes end-to-end so the LLM
                sees the modified prompt / input / tool overrides.
              </>
            ) : (
              "Select a span first."
            )}
          </DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <Label htmlFor="fork-determinism">LLM calls</Label>
            <select
              id="fork-determinism"
              value={determinism}
              onChange={(e) => setDeterminism(e.target.value as ReplayDeterminism)}
              className={SELECT_CLASS}
            >
              <option value="live">Live (call the model)</option>
              <option value="recorded">Recorded (reuse the trace's responses)</option>
              <option value="deterministic">Deterministic (live, temperature 0)</option>
            </select>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="fork-on-miss">When a recorded response is missing</Label>
            <select
              id="fork-on-miss"
              value={onMiss}
              onChange={(e) => setOnMiss(e.target.value as ReplayOnMiss)}
              disabled={determinism !== "recorded"}
              title={
                determinism !== "recorded"
                  ? "Only applies to Recorded mode"
                  : undefined
              }
              className={cn(SELECT_CLASS, "disabled:cursor-not-allowed disabled:opacity-50")}
            >
              <option value="live">Call the model live (billed)</option>
              <option value="error">Fail the rerun</option>
            </select>
          </div>
        </div>

        <Tabs defaultValue="prompt">
          <TabsList className="w-full">
            <TabsTrigger value="prompt" className="flex-1">
              Prompt
            </TabsTrigger>
            <TabsTrigger value="input" className="flex-1">
              Input
            </TabsTrigger>
            <TabsTrigger value="tool" className="flex-1">
              Tools
            </TabsTrigger>
            <TabsTrigger value="params" className="flex-1">
              LLM params
            </TabsTrigger>
          </TabsList>

          <TabsContent value="prompt" className="space-y-2">
            <Label htmlFor="fork-prompt">System prompt override</Label>
            <Textarea
              id="fork-prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="New system prompt, or leave blank to keep the original."
              rows={6}
              className="font-mono text-xs"
            />
          </TabsContent>

          <TabsContent value="input" className="space-y-2">
            <Label htmlFor="fork-input">Input override (JSON object or list of parts)</Label>
            <Textarea
              id="fork-input"
              value={inputJson}
              onChange={(e) => setInputJson(e.target.value)}
              placeholder='{"input": "new user prompt"}  or  [{"type":"text","text":"..."},{"type":"image","data_base64":"..."}]'
              rows={6}
              className="font-mono text-xs"
            />
            <div className="flex flex-wrap items-center gap-2">
              <Label htmlFor="fork-replace-image" className="text-xs">
                Replace image:
              </Label>
              <Input
                id="fork-replace-image"
                type="file"
                accept="image/*,application/pdf"
                onChange={async (e) => {
                  const file = e.target.files?.[0];
                  if (!file) return;
                  try {
                    const data_base64 = await fileToBase64(file);
                    const isPdf = file.type === "application/pdf";
                    const parts: unknown[] = [
                      { type: "text", text: "Replaced via fork dialog" },
                      isPdf
                        ? { type: "pdf", data_base64 }
                        : {
                            type: "image",
                            data_base64,
                            media_type: file.type || "image/png",
                          },
                    ];
                    setInputJson(JSON.stringify(parts, null, 2));
                  } catch (err) {
                    toast.error(`Failed to read file: ${String(err)}`);
                  } finally {
                    e.target.value = "";
                  }
                }}
                className="text-xs"
              />
            </div>
            <p className="text-xs text-muted-foreground">
              Blank = no change. JSON object replaces the prompt (legacy
              form); JSON list replaces with a multimodal payload — each
              part is one of <code>{`{"type":"text",...}`}</code>,{" "}
              <code>{`{"type":"image","data_base64":"..."}`}</code>, or{" "}
              <code>{`{"type":"pdf","data_base64":"..."}`}</code>.
            </p>
          </TabsContent>

          <TabsContent value="tool" className="space-y-2">
            {tools.length > 0 ? (
              <>
                <p className="text-xs text-muted-foreground">
                  A rerun runs only the tools you stub or allow. A blocked tool
                  is not called: the model gets an error result instead.
                </p>
                <ul className="max-h-72 divide-y overflow-y-auto rounded-md border">
                  {tools.map((t) => (
                    <ToolRow
                      key={t.name}
                      tool={t}
                      mode={modeOf(t.name)}
                      stubText={stubTexts[t.name] ?? ""}
                      onModeChange={(m) =>
                        setToolModes((prev) => ({ ...prev, [t.name]: m }))
                      }
                      onStubChange={(v) =>
                        setStubTexts((prev) => ({ ...prev, [t.name]: v }))
                      }
                    />
                  ))}
                </ul>
              </>
            ) : (
              <>
                <p className="text-xs text-muted-foreground">
                  The trace recorded no tool list for this agent. You can
                  still stub a tool by name.
                </p>
                <div className="space-y-1.5">
                  <Label htmlFor="fork-tool-name">Tool name</Label>
                  <Input
                    id="fork-tool-name"
                    value={toolName}
                    onChange={(e) => setToolName(e.target.value)}
                    placeholder="lookup_order"
                  />
                  <p className="text-[10px] text-muted-foreground">
                    The rerun installs a stub <code>FunctionTool</code> with
                    this name; the LLM receives the JSON below in place of
                    calling the original tool.
                  </p>
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="fork-tool">Tool response override (JSON)</Label>
                  <Textarea
                    id="fork-tool"
                    value={toolJson}
                    onChange={(e) => setToolJson(e.target.value)}
                    placeholder='{"result": "canned value"}'
                    rows={6}
                    className="font-mono text-xs"
                  />
                </div>
              </>
            )}
          </TabsContent>

          <TabsContent value="params" className="grid grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <Label htmlFor="fork-temp">Temperature</Label>
              <Input
                id="fork-temp"
                type="number"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(e.target.value)}
                placeholder="0.7"
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="fork-maxtok">Max tokens</Label>
              <Input
                id="fork-maxtok"
                type="number"
                value={maxTokens}
                onChange={(e) => setMaxTokens(e.target.value)}
                placeholder="1024"
              />
            </div>
          </TabsContent>
        </Tabs>

        {tools.length > 0 && (
          <p data-testid="fork-tool-summary" className="text-xs text-muted-foreground">
            Tools: {counts.block} blocked · {counts.stub} stubbed ·{" "}
            {counts.live} live (see the Tools tab).
          </p>
        )}

        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={pending}>
            Cancel
          </Button>
          <Button
            onClick={handleRerun}
            disabled={pending || !step}
            title="Re-runs the agent from the top with your modifications applied. v2 will support true mid-trace resume."
          >
            {pending ? (
              <>
                <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                Rerunning…
              </>
            ) : (
              <>
                <Play className="mr-1.5 h-3.5 w-3.5" />
                Rerun with modifications
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

const TOOL_MODES: { value: ToolMode; label: string }[] = [
  { value: "block", label: "Block" },
  { value: "stub", label: "Stub" },
  { value: "live", label: "Live" },
];

function ToolRow({
  tool,
  mode,
  stubText,
  onModeChange,
  onStubChange,
}: {
  tool: ReplayToolChoice;
  mode: ToolMode;
  stubText: string;
  onModeChange: (mode: ToolMode) => void;
  onStubChange: (value: string) => void;
}) {
  const liveReason = liveBlockedReason(tool);
  return (
    <li className="space-y-2 px-3 py-2" data-testid={`fork-tool-${tool.name}`}>
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <span className="block truncate font-mono text-xs">{tool.name}</span>
          <span className="font-mono text-[10px] text-muted-foreground">
            {tool.tool_type}
          </span>
        </div>
        <div
          role="group"
          aria-label={`What the rerun does with ${tool.name}`}
          className="flex shrink-0 items-center gap-1 rounded-md border bg-card p-0.5"
        >
          {TOOL_MODES.map((m) => {
            const disabled = m.value === "live" && liveReason !== null;
            return (
              <button
                key={m.value}
                type="button"
                aria-pressed={mode === m.value}
                aria-label={`${m.label} ${tool.name}`}
                disabled={disabled}
                title={disabled ? liveReason ?? undefined : undefined}
                onClick={() => onModeChange(m.value)}
                className={cn(
                  "rounded px-2 py-1 text-[11px] font-mono font-medium transition-colors",
                  "disabled:cursor-not-allowed disabled:opacity-40",
                  mode === m.value
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted"
                )}
              >
                {m.label}
              </button>
            );
          })}
        </div>
      </div>
      {liveReason && (
        <p className="text-[10px] text-muted-foreground">{liveReason}</p>
      )}
      {mode === "live" && tool.endpoint_from_trace && (
        <p className="text-[10px] text-fa-warning">
          Live calls the endpoint recorded in the trace, for real.
        </p>
      )}
      {mode === "stub" && (
        <Textarea
          aria-label={`Stub response for ${tool.name}`}
          value={stubText}
          onChange={(e) => onStubChange(e.target.value)}
          placeholder='{"result": "canned value"}  (JSON, or plain text)'
          rows={3}
          className="font-mono text-xs"
        />
      )}
    </li>
  );
}
