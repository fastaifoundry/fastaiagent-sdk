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
import type { RerunResult, ReplayStep } from "@/lib/types";
import { useForkAtStep, useModifyFork, useRerunFork } from "@/hooks/use-replay";

interface Props {
  traceId: string;
  step: ReplayStep | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onRerunComplete: (fork_id: string, result: RerunResult) => void;
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
}: Props) {
  const [prompt, setPrompt] = useState("");
  const [inputJson, setInputJson] = useState("");
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
    setToolJson("");
    setTemperature("");
    setMaxTokens("");
  };

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
      if (parsedTool && Object.keys(parsedTool).length > 0) mods.tool_response = parsedTool;

      const config: Record<string, unknown> = {};
      if (temperature !== "") config.temperature = Number(temperature);
      if (maxTokens !== "") config.max_tokens = Number(maxTokens);
      if (Object.keys(config).length > 0) mods.config = config;

      if (Object.keys(mods).length > 0) {
        await modify.mutateAsync({ forkId: fork_id, mods });
      }

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
          <DialogTitle>Fork and rerun</DialogTitle>
          <DialogDescription>
            {step ? (
              <>
                Forking at step <span className="font-mono">{step.step}</span>:{" "}
                <span className="font-mono text-foreground">{step.span_name}</span>
              </>
            ) : (
              "Select a span first."
            )}
          </DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="prompt">
          <TabsList className="w-full">
            <TabsTrigger value="prompt" className="flex-1">
              Prompt
            </TabsTrigger>
            <TabsTrigger value="input" className="flex-1">
              Input
            </TabsTrigger>
            <TabsTrigger value="tool" className="flex-1">
              Tool response
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
            <Label htmlFor="fork-tool">Tool response override (JSON object)</Label>
            <Textarea
              id="fork-tool"
              value={toolJson}
              onChange={(e) => setToolJson(e.target.value)}
              placeholder='{"result": "canned value"}'
              rows={6}
              className="font-mono text-xs"
            />
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

        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={pending}>
            Cancel
          </Button>
          <Button onClick={handleRerun} disabled={pending || !step}>
            {pending ? (
              <>
                <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                Rerunning…
              </>
            ) : (
              <>
                <Play className="mr-1.5 h-3.5 w-3.5" />
                Rerun from this step
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
