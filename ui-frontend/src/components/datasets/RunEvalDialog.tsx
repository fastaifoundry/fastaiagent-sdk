import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { AlertTriangle, Loader2, Play, Plus, X } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useRunEval, useRunEvalOptions } from "@/hooks/use-datasets";
import { formatRate } from "@/lib/evalDisplay";
import type {
  DatasetRunEvalResult,
  JudgeModelOption,
  JudgeScale,
  RunEvalBody,
  RunEvalOptions,
  ScorerFamily,
  ScorerOption,
  ScorerParam,
  ScorerSpec,
} from "@/lib/types";

/**
 * Run eval from the dataset page (1.89.0): pick a registered agent or a
 * dataset check (inputs echoed back), pick scorers with their parameters and
 * judge model, see what the dataset cannot feed before running, and get the
 * run's numbers inline. Everything comes from GET .../run-eval/options; the
 * server validates the request again and its 422 `detail` is shown verbatim.
 */

interface Props {
  datasetName: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const SELECT_CLASS =
  "h-9 w-full rounded-md border border-input bg-background px-2 text-sm disabled:cursor-not-allowed disabled:opacity-50";

const FAMILIES: { family: ScorerFamily; label: string }[] = [
  { family: "deterministic", label: "Deterministic" },
  { family: "llm", label: "LLM judges" },
  { family: "trajectory", label: "Trajectory" },
  { family: "rag", label: "RAG" },
];

const SCALES: JudgeScale[] = ["binary", "0-1", "1-5", "1-10"];

/** Catalog entries that are their own spec type, never `{type: "builtin"}`. */
const OWN_SPEC_TYPES = new Set(["llm_judge", "geval"]);

/** Deterministic scorers that read the run, not its text. */
const NON_TEXT_SCORERS = new Set(["latency", "cost_under"]);

const NEEDS_AGENT_REASON = "needs an agent: it measures the agent's run";

interface RubricRow {
  score: string;
  description: string;
}

interface ScorerConfig {
  /** Raw input values, by param name. */
  params: Record<string, string>;
  /** Index into `judge_models`. */
  model: number;
  steps: string[];
  rubric: RubricRow[];
}

function defaultJudgeModel(models: JudgeModelOption[]): number {
  const mini = models.findIndex((m) => m.model === "gpt-4o-mini");
  return mini >= 0 ? mini : 0;
}

function initialConfig(scorer: ScorerOption, models: JudgeModelOption[]): ScorerConfig {
  return {
    params: Object.fromEntries(
      scorer.params.map((p) => [p.name, p.default == null ? "" : String(p.default)])
    ),
    model: defaultJudgeModel(models),
    steps: [],
    rubric: [],
  };
}

function disabledReason(scorer: ScorerOption, mode: "agent" | "echo"): string | null {
  if (!scorer.available) return scorer.unavailable_reason ?? "not available here";
  if (mode === "echo" && scorer.needs_agent) return NEEDS_AGENT_REASON;
  return null;
}

function paramValue(p: ScorerParam, raw: string): string | number {
  if (p.type === "string") return raw;
  const n = Number(raw);
  // Not a number: send it as typed and let the server say why.
  return Number.isFinite(n) ? n : raw;
}

function numberOr(raw: string | undefined, fallback: number): number {
  const n = Number(raw);
  return raw != null && raw.trim() !== "" && Number.isFinite(n) ? n : fallback;
}

/** One scorer as the request carries it. */
function buildScorerSpec(
  scorer: ScorerOption,
  config: ScorerConfig,
  judgeModels: JudgeModelOption[]
): string | ScorerSpec {
  const model = scorer.judge ? judgeModels[config.model] : undefined;
  const modelPart = model ? { model: { provider: model.provider, model: model.model } } : {};
  if (OWN_SPEC_TYPES.has(scorer.name)) {
    const thresholdDefault = scorer.params.find((p) => p.name === "threshold")?.default;
    const scaleDefault = scorer.params.find((p) => p.name === "scale")?.default;
    const base = {
      criteria: (config.params.criteria ?? "").trim(),
      scale: (config.params.scale || String(scaleDefault ?? "binary")) as JudgeScale,
      threshold: numberOr(
        config.params.threshold,
        typeof thresholdDefault === "number" ? thresholdDefault : 0.5
      ),
      ...modelPart,
    };
    if (scorer.name === "llm_judge") return { type: "llm_judge", ...base };
    const steps = config.steps.map((s) => s.trim()).filter(Boolean);
    const rubric = config.rubric
      .filter((r) => r.description.trim())
      .map((r): [number, string] => [Number(r.score), r.description.trim()]);
    return {
      type: "geval",
      ...base,
      ...(steps.length ? { evaluation_steps: steps } : {}),
      ...(rubric.length ? { rubric } : {}),
    };
  }
  const params: Record<string, string | number> = {};
  for (const p of scorer.params) {
    const raw = config.params[p.name] ?? "";
    if (raw.trim() === "") continue;
    params[p.name] = paramValue(p, raw);
  }
  if (Object.keys(params).length === 0 && !model) return scorer.name;
  return {
    type: "builtin",
    name: scorer.name,
    ...(Object.keys(params).length ? { params } : {}),
    ...modelPart,
  };
}

/** What a run of these scorers on this dataset will not show plainly. */
function runWarnings(
  options: RunEvalOptions,
  active: ScorerOption[],
  mode: "agent" | "echo"
): string[] {
  const f = options.dataset_features;
  const out: string[] = [];
  if (mode === "echo") {
    out.push("Dataset check: inputs are echoed back; the result does not measure a model");
  }
  for (const s of active) {
    for (const need of s.needs) {
      const missing = s.rows_missing[need] ?? 0;
      if (missing > 0) {
        out.push(
          `${s.name} needs ${need}; ${missing} of ${f.rows} row${f.rows === 1 ? "" : "s"} ` +
            "lack it: those cases are n/a"
        );
      }
    }
  }
  if (
    f.multimodal > 0 &&
    active.some((s) => s.family === "deterministic" && !NON_TEXT_SCORERS.has(s.name))
  ) {
    out.push(
      `${f.multimodal} row${f.multimodal === 1 ? " has" : "s have"} images or PDFs; ` +
        "text scorers may not apply to them"
    );
  }
  const judges = active.filter((s) => s.judge).length;
  if (judges > 0) {
    out.push(
      `~${f.rows * judges} judge calls (${f.rows} row${f.rows === 1 ? "" : "s"} × ` +
        `${judges} judge scorer${judges === 1 ? "" : "s"})`
    );
  }
  return out;
}

export function RunEvalDialog({ datasetName, open, onOpenChange }: Props) {
  const options = useRunEvalOptions(datasetName, open);
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Run eval</DialogTitle>
          <DialogDescription>
            Score every case in <code>{datasetName}.jsonl</code>. The run is saved and
            listed under Evals.
          </DialogDescription>
        </DialogHeader>
        {options.isLoading ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Loading options…
          </div>
        ) : options.error || !options.data ? (
          <ErrorAlert
            message={
              options.error instanceof Error ? options.error.message : "Could not load options"
            }
          />
        ) : (
          <RunEvalForm
            datasetName={datasetName}
            options={options.data}
            onClose={() => onOpenChange(false)}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}

function ErrorAlert({ message }: { message: string }) {
  return (
    <div
      role="alert"
      className="whitespace-pre-wrap rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-600"
    >
      {message}
    </div>
  );
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="text-xs uppercase tracking-widest text-muted-foreground">{children}</div>
  );
}

function RunEvalForm({
  datasetName,
  options,
  onClose,
}: {
  datasetName: string;
  options: RunEvalOptions;
  onClose: () => void;
}) {
  const runEval = useRunEval(datasetName);
  const supported = options.agents.filter((a) => a.supported);
  const judgeModels = options.judge_models;

  const [mode, setMode] = useState<"agent" | "echo">(supported.length ? "agent" : "echo");
  const [agentName, setAgentName] = useState(supported[0]?.name ?? "");
  const [selected, setSelected] = useState<Set<string>>(() => {
    const exact = options.scorers.find((s) => s.name === "exact_match");
    return new Set(exact?.available ? ["exact_match"] : []);
  });
  const [configs, setConfigs] = useState<Record<string, ScorerConfig>>(() =>
    Object.fromEntries(options.scorers.map((s) => [s.name, initialConfig(s, judgeModels)]))
  );
  const [runName, setRunName] = useState("");
  const [result, setResult] = useState<DatasetRunEvalResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const active = options.scorers.filter(
    (s) => selected.has(s.name) && !disabledReason(s, mode)
  );
  const warnings = runWarnings(options, active, mode);

  const toggle = (name: string, on: boolean) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (on) next.add(name);
      else next.delete(name);
      return next;
    });

  const patchConfig = (name: string, patch: Partial<ScorerConfig>) =>
    setConfigs((prev) => ({ ...prev, [name]: { ...prev[name]!, ...patch } }));

  const validate = (): string | null => {
    if (mode === "agent" && !agentName) return "Pick an agent, or run a dataset check.";
    if (active.length === 0) return "Pick at least one scorer.";
    for (const s of active) {
      const cfg = configs[s.name]!;
      for (const p of s.params) {
        if (p.required && !(cfg.params[p.name] ?? "").trim()) {
          return `${s.name} needs ${p.name}.`;
        }
      }
      if (s.name === "geval") {
        const bad = cfg.rubric.find(
          (r) => r.description.trim() && !Number.isFinite(Number(r.score || "x"))
        );
        if (bad) return "geval: every rubric row needs a numeric score.";
      }
    }
    return null;
  };

  const handleRun = async () => {
    setError(null);
    setResult(null);
    const problem = validate();
    if (problem) {
      setError(problem);
      return;
    }
    const scorers = active.map((s) => buildScorerSpec(s, configs[s.name]!, judgeModels));
    const body: RunEvalBody =
      mode === "echo"
        ? { mode: "echo", scorers }
        : { agent_name: agentName, mode: "agent", scorers };
    if (runName.trim()) body.run_name = runName.trim();
    try {
      setResult(await runEval.mutateAsync(body));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Run failed");
    }
  };

  const pending = runEval.isPending;

  return (
    <>
      <div className="space-y-5">
        <fieldset className="space-y-2" disabled={pending}>
          <SectionLabel>Target</SectionLabel>
          <label
            className={`flex items-center gap-2 text-sm ${supported.length ? "" : "opacity-50"}`}
          >
            <input
              type="radio"
              name="run-eval-target"
              value="agent"
              checked={mode === "agent"}
              disabled={supported.length === 0}
              onChange={() => setMode("agent")}
            />
            Agent
          </label>
          {mode === "agent" && (
            <div className="ml-6">
              <select
                aria-label="Agent"
                value={agentName}
                onChange={(e) => setAgentName(e.target.value)}
                className={SELECT_CLASS}
              >
                {options.agents.map((a) => (
                  <option key={a.name} value={a.name} disabled={!a.supported}>
                    {a.name} ({a.kind}){a.supported ? "" : ` — ${a.reason ?? "not supported"}`}
                  </option>
                ))}
              </select>
            </div>
          )}
          {supported.length === 0 && (
            <p className="ml-6 text-xs text-muted-foreground">
              {options.agents.length === 0 ? (
                <>
                  Register an agent:{" "}
                  <code className="font-mono">fastaiagent ui --agent app.py:my_agent</code>
                </>
              ) : (
                options.agents.map((a) => `${a.name}: ${a.reason ?? "not supported"}`).join("; ")
              )}
            </p>
          )}
          <label className="flex items-center gap-2 text-sm">
            <input
              type="radio"
              name="run-eval-target"
              value="echo"
              checked={mode === "echo"}
              onChange={() => setMode("echo")}
            />
            Dataset check (echo)
          </label>
          {mode === "echo" && (
            <p className="ml-6 text-xs text-muted-foreground">
              Echoes each input back as the output. It tests the dataset, not a model.
            </p>
          )}
        </fieldset>

        <fieldset className="space-y-3" disabled={pending}>
          <SectionLabel>Scorers</SectionLabel>
          {FAMILIES.map(({ family, label }) => {
            const group = options.scorers.filter((s) => s.family === family);
            if (group.length === 0) return null;
            return (
              <div key={family} className="space-y-1.5" data-testid={`scorers-${family}`}>
                <div className="text-[11px] font-medium text-muted-foreground">{label}</div>
                {group.map((s) => {
                  const reason = disabledReason(s, mode);
                  const checked = selected.has(s.name) && !reason;
                  return (
                    <div key={s.name} className="rounded-md border bg-card px-3 py-2">
                      <div className="flex items-start gap-2">
                        <Checkbox
                          id={`scorer-${s.name}`}
                          aria-label={s.name}
                          className="mt-0.5"
                          checked={checked}
                          disabled={!!reason}
                          onCheckedChange={(v) => toggle(s.name, v === true)}
                        />
                        <label
                          htmlFor={`scorer-${s.name}`}
                          className={`min-w-0 flex-1 text-sm ${reason ? "opacity-60" : "cursor-pointer"}`}
                        >
                          <span className="font-mono text-xs">{s.name}</span>{" "}
                          <span className="text-xs text-muted-foreground">{s.description}</span>
                          {reason && (
                            <span className="block text-[11px] text-muted-foreground">
                              {reason}
                            </span>
                          )}
                        </label>
                      </div>
                      {checked && (
                        <ScorerFields
                          scorer={s}
                          config={configs[s.name]!}
                          judgeModels={judgeModels}
                          onChange={(patch) => patchConfig(s.name, patch)}
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            );
          })}
        </fieldset>

        <div className="space-y-1">
          <Label htmlFor="run-eval-name" className="text-xs uppercase tracking-widest text-muted-foreground">
            Run name <span className="normal-case tracking-normal">(optional)</span>
          </Label>
          <Input
            id="run-eval-name"
            value={runName}
            disabled={pending}
            onChange={(e) => setRunName(e.target.value)}
            placeholder={mode === "echo" ? `dataset-check-${datasetName}` : `editor-${datasetName}`}
          />
        </div>

        {warnings.length > 0 && (
          <ul
            data-testid="run-eval-warnings"
            className="space-y-1 rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-xs text-amber-900 dark:text-amber-200"
          >
            {warnings.map((w) => (
              <li key={w} className="flex items-start gap-1.5">
                <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
                <span>{w}</span>
              </li>
            ))}
          </ul>
        )}

        {error && <ErrorAlert message={error} />}
        {result && <RunResult result={result} />}
      </div>

      <DialogFooter>
        <Button variant="ghost" onClick={onClose}>
          Close
        </Button>
        <Button onClick={handleRun} disabled={pending}>
          {pending ? (
            <>
              <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
              Running…
            </>
          ) : (
            <>
              <Play className="mr-1.5 h-3.5 w-3.5" />
              {result ? "Run again" : "Run"}
            </>
          )}
        </Button>
      </DialogFooter>
    </>
  );
}

function ScorerFields({
  scorer,
  config,
  judgeModels,
  onChange,
}: {
  scorer: ScorerOption;
  config: ScorerConfig;
  judgeModels: JudgeModelOption[];
  onChange: (patch: Partial<ScorerConfig>) => void;
}) {
  const setParam = (name: string, value: string) =>
    onChange({ params: { ...config.params, [name]: value } });
  const fieldId = (param: string) => `scorer-${scorer.name}-${param}`;
  const hasFields =
    scorer.params.length > 0 || (scorer.judge && judgeModels.length > 0) || scorer.name === "geval";
  if (!hasFields) return null;

  return (
    <div className="mt-2 space-y-2 border-t pt-2 pl-6">
      {scorer.params.map((p) => (
        <div key={p.name} className="space-y-1">
          <Label htmlFor={fieldId(p.name)} className="text-xs">
            {p.name}
            {p.required && <span className="text-destructive"> *</span>}
            {p.description && p.name !== "scale" && (
              <span className="ml-2 font-normal text-muted-foreground">{p.description}</span>
            )}
          </Label>
          {p.name === "criteria" ? (
            <Textarea
              id={fieldId(p.name)}
              aria-label={`${scorer.name} ${p.name}`}
              rows={3}
              value={config.params[p.name] ?? ""}
              onChange={(e) => setParam(p.name, e.target.value)}
              placeholder="What a good answer does."
            />
          ) : p.name === "scale" ? (
            <select
              id={fieldId(p.name)}
              aria-label={`${scorer.name} ${p.name}`}
              value={config.params[p.name] ?? ""}
              onChange={(e) => setParam(p.name, e.target.value)}
              className={SELECT_CLASS}
            >
              {SCALES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          ) : (
            <Input
              id={fieldId(p.name)}
              aria-label={`${scorer.name} ${p.name}`}
              type={p.type === "string" ? "text" : "number"}
              min={p.minimum ?? undefined}
              max={p.maximum ?? undefined}
              step={p.type === "integer" ? 1 : p.type === "number" ? "any" : undefined}
              value={config.params[p.name] ?? ""}
              onChange={(e) => setParam(p.name, e.target.value)}
              className={p.type === "string" ? "font-mono text-xs" : undefined}
            />
          )}
        </div>
      ))}

      {scorer.name === "geval" && <GEvalFields config={config} onChange={onChange} />}

      {scorer.judge && judgeModels.length > 0 && (
        <div className="space-y-1">
          <Label htmlFor={fieldId("model")} className="text-xs">
            Judge model
          </Label>
          <select
            id={fieldId("model")}
            aria-label={`${scorer.name} judge model`}
            value={String(config.model)}
            onChange={(e) => onChange({ model: Number(e.target.value) })}
            className={SELECT_CLASS}
          >
            {judgeModels.map((m, i) => (
              <option key={`${m.provider}-${m.model}`} value={String(i)}>
                {m.provider} · {m.model}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}

function GEvalFields({
  config,
  onChange,
}: {
  config: ScorerConfig;
  onChange: (patch: Partial<ScorerConfig>) => void;
}) {
  const setStep = (i: number, value: string) =>
    onChange({ steps: config.steps.map((s, j) => (j === i ? value : s)) });
  const setRubric = (i: number, patch: Partial<RubricRow>) =>
    onChange({ rubric: config.rubric.map((r, j) => (j === i ? { ...r, ...patch } : r)) });

  return (
    <>
      <div className="space-y-1">
        <div className="text-xs font-medium">
          Evaluation steps{" "}
          <span className="font-normal text-muted-foreground">
            (optional; the judge writes its own when empty)
          </span>
        </div>
        {config.steps.map((step, i) => (
          <div key={i} className="flex items-center gap-1.5">
            <Input
              aria-label={`geval step ${i + 1}`}
              value={step}
              onChange={(e) => setStep(i, e.target.value)}
              placeholder={`Step ${i + 1}`}
            />
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-8 w-8 shrink-0"
              aria-label={`Remove step ${i + 1}`}
              onClick={() => onChange({ steps: config.steps.filter((_, j) => j !== i) })}
            >
              <X className="h-3.5 w-3.5" />
            </Button>
          </div>
        ))}
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => onChange({ steps: [...config.steps, ""] })}
        >
          <Plus className="mr-1 h-3.5 w-3.5" />
          Add step
        </Button>
      </div>
      <div className="space-y-1">
        <div className="text-xs font-medium">
          Rubric <span className="font-normal text-muted-foreground">(optional; score and what it means)</span>
        </div>
        {config.rubric.map((row, i) => (
          <div key={i} className="flex items-center gap-1.5">
            <Input
              aria-label={`geval rubric ${i + 1} score`}
              type="number"
              step="any"
              className="w-20 shrink-0"
              value={row.score}
              onChange={(e) => setRubric(i, { score: e.target.value })}
            />
            <Input
              aria-label={`geval rubric ${i + 1} description`}
              value={row.description}
              onChange={(e) => setRubric(i, { description: e.target.value })}
              placeholder="What this score means"
            />
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-8 w-8 shrink-0"
              aria-label={`Remove rubric row ${i + 1}`}
              onClick={() => onChange({ rubric: config.rubric.filter((_, j) => j !== i) })}
            >
              <X className="h-3.5 w-3.5" />
            </Button>
          </div>
        ))}
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => onChange({ rubric: [...config.rubric, { score: "", description: "" }] })}
        >
          <Plus className="mr-1 h-3.5 w-3.5" />
          Add rubric row
        </Button>
      </div>
    </>
  );
}

function RunResult({ result }: { result: DatasetRunEvalResult }) {
  const cases = result.cases ?? {
    passed: result.pass_count,
    failed: result.fail_count,
    errored: result.errored_count ?? 0,
    pass_rate: result.pass_rate,
  };
  const checks = result.checks;
  const echo = result.mode === "echo";
  const checkParts = checks
    ? [
        `${checks.passed} passed`,
        `${checks.failed} failed`,
        checks.errored ? `${checks.errored} errored` : null,
        checks.skipped ? `${checks.skipped} n/a` : null,
      ].filter(Boolean)
    : [];
  return (
    <div
      role="status"
      data-testid="run-eval-result"
      className="space-y-1 rounded-md border bg-muted/40 px-3 py-2 text-sm"
    >
      <div className="font-medium">{echo ? "Dataset check finished" : "Eval finished"}</div>
      <div>
        Cases {formatRate(cases.pass_rate)}: {cases.passed} passed · {cases.failed} failed ·{" "}
        {cases.errored} errored
      </div>
      {checks && (
        <div className="text-muted-foreground">
          Checks {formatRate(checks.pass_rate)}: {checkParts.join(" · ")}
        </div>
      )}
      {echo && result.note && <p className="text-xs text-muted-foreground">{result.note}</p>}
      <Link
        to={`/evals/${encodeURIComponent(result.run_id)}`}
        className="inline-block text-sm font-medium text-primary underline-offset-4 hover:underline"
      >
        Open run
      </Link>
    </div>
  );
}
