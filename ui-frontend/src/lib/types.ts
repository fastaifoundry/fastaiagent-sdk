/**
 * Shared TypeScript types mirroring the FastAPI server's Pydantic models.
 */

// Root-span classifications the trace list can show. The first four are
// runners; "guardrail" / "evaluator" are OpenInference span kinds that reach
// us as a trace root when a foreign runtime emits a standalone check or score
// (fastaiagent.emit_guardrail / emit_evaluation).
export type RunnerType =
  | "agent"
  | "chain"
  | "swarm"
  | "supervisor"
  | "guardrail"
  | "evaluator";

// The multi-step workflow topologies — a strict subset of RunnerType.
export type WorkflowRunnerType = "chain" | "swarm" | "supervisor";

export interface TraceRow {
  trace_id: string;
  name: string;
  start_time: string;
  end_time: string | null;
  status: string;
  span_count: number;
  duration_ms: number | null;
  agent_name: string | null;
  thread_id: string | null;
  total_cost_usd: number | null;
  total_tokens: number | null;
  runner_type: RunnerType;
  runner_name: string | null;
  // Source framework drawn from ``fastaiagent.framework`` on the root
  // span. Free-text — any value the harness or a custom instrumentation
  // has stamped survives a trip through the API as-is.
  framework?: string | null;
  // Where the run originated, from ``fastaiagent.source`` on any span of the
  // trace: "replay", "playground", "learn", "eval", "simulation", "optimize",
  // "kb_search". Null = production. Filter via ?source= or
  // ?exclude_experiments=true (1.89.0).
  source?: string | null;
  // Whether this trace is starred (present in trace_favorites). Drives the
  // filled star in the Traces list.
  favorited?: boolean;
}

export interface TracesPage {
  rows: TraceRow[];
  total: number;
  page: number;
  page_size: number;
}

export interface SpanRow {
  span_id: string;
  trace_id: string;
  parent_span_id: string | null;
  name: string;
  start_time: string;
  end_time: string;
  status: string;
  attributes: Record<string, unknown>;
  events: SpanEvent[];
  /**
   * 1.89.0 — for a ``guardrail.*`` span, what the check decided, joined from
   * its local event row. A guardrail span's OTel status is ERROR for any
   * non-pass verdict (a ``warn`` included), so read this, not ``status``.
   */
  guardrail?: SpanGuardrailVerdict | null;
}

export interface SpanGuardrailVerdict {
  event_id: string;
  outcome: string | null;
  effect: GuardrailEffect | null;
  blocking: boolean | null;
  final_reason: string | null;
}

export interface SpanEvent {
  name: string;
  timestamp: string;
  attributes?: Record<string, unknown>;
}

export interface SpanTreeNode {
  span: SpanRow;
  children: SpanTreeNode[];
}

export interface TraceDetail {
  trace_id: string;
  name: string;
  status: string;
  start_time: string;
  end_time: string;
  agent_name: string | null;
  thread_id: string | null;
  total_cost_usd: number | null;
  total_tokens: number | null;
  span_count: number;
  runner_type: RunnerType;
  runner_name: string | null;
  spans: SpanRow[];
}

export interface ReplayStep {
  step: number;
  span_name: string;
  span_id: string;
  input: Record<string, unknown>;
  output: Record<string, unknown>;
  attributes: Record<string, unknown>;
  timestamp: string;
}

/** What a rerun did with one tool (1.89.0). */
export type ToolDisposition =
  | "blocked"
  | "live"
  | "stub"
  | "override"
  | "no_function";

export type ReplayDeterminism = "live" | "recorded" | "deterministic";
export type ReplayOnMiss = "live" | "error";
export type ReplayToolPolicy = "block" | "error" | "live";

/**
 * One of the original agent's tools, and what a rerun could do with it
 * (``GET /api/replay/{trace_id}`` → ``tools``, 1.89.0).
 */
export interface ReplayToolChoice {
  name: string;
  tool_type: string;
  origin: string | null;
  replay_class: string | null;
  /** This process has the function (a registered runner's tool, or the registry). */
  live_available: boolean;
  /** A REST/MCP tool whose endpoint the trace recorded; Live calls it for real. */
  endpoint_from_trace: boolean;
  /** Its auth headers were redacted, so a live call from the trace can't authenticate. */
  credentials_redacted: boolean;
}

export interface ReplayPayload {
  steps: ReplayStep[];
  tools?: ReplayToolChoice[];
}

export interface RerunResult {
  fork_id: string;
  new_trace_id: string | null;
  new_output: unknown;
  original_output: unknown;
  steps_executed: number;
  // 1.89.0 — optional so an older server's payload still type-checks.
  source?: "replay";
  determinism?: ReplayDeterminism;
  tool_policy?: ReplayToolPolicy;
  tool_dispositions?: Record<string, ToolDisposition>;
  /** Calls the rerun's model made to a blocked tool, by name. */
  blocked_tool_calls?: Record<string, number>;
  /** Whether the rerun's spans may leave the machine (off by default). */
  exported?: boolean;
}

/** One aligned tool call whose outcome changed between the runs (1.89.0). */
export interface ToolDiff {
  original_step: number | null;
  new_step: number | null;
  name: string;
  original_status: string | null;
  new_status: string | null;
  original_result: unknown;
  new_result: unknown;
  disposition: ToolDisposition | string | null;
}

export interface ComparisonResult {
  original_steps: ReplayStep[];
  new_steps: ReplayStep[];
  /**
   * First step index where original and rerun diverge. ``null`` means
   * the two traces matched step-for-step (or — when ``compare_status``
   * is ``"rerun_failed"`` — that the rerun trace couldn't be loaded
   * so divergence couldn't be computed).
   *
   * v1.14: was hardcoded to ``fork_point`` in v1.13; now computed from
   * span-by-span comparison.
   */
  diverged_at: number | null;
  /**
   * Distinguishes "rerun loaded fine, this is the real divergence" from
   * "rerun trace couldn't be loaded — divergence is unknown".
   *
   * Added in v1.14. Backwards-compatible: callers that don't read this
   * field still get the correct ``diverged_at`` value for the ``"ok"``
   * case.
   */
  compare_status?: "ok" | "rerun_failed";
  // 1.89.0, all additive. ``diverged_at`` indexes ``original_steps``; only
  // the run's own turns are aligned (memory and nested spans are skipped).
  /** The same divergence, as an index into ``new_steps``. */
  new_diverged_at?: number | null;
  /** ``original_steps`` left out of the alignment (memory, nested spans). */
  skipped_steps?: number[];
  new_skipped_steps?: number[];
  /** ``[original_index, new_index]`` pairs, in order; null where one side ran fewer turns. */
  aligned?: [number | null, number | null][];
  /** Aligned tool calls whose status or result differ. */
  tool_diffs?: ToolDiff[];
  /** Index into ``original_steps`` of the first tool call that differs. */
  tools_diverged_at?: number | null;
}

// ---------------------------------------------------------------------------
// Trace Comparison (Sprint 3)
// ---------------------------------------------------------------------------

export type CompareMatchKind =
  | "same"
  | "slower"
  | "faster"
  | "different_output"
  | "new_in_a"
  | "new_in_b";

export interface CompareSpanSummary {
  span_id: string;
  name: string;
  status: string;
  start_time: string;
  end_time: string;
  duration_ms: number | null;
}

export interface CompareAlignmentRow {
  index: number;
  span_a: CompareSpanSummary | null;
  span_b: CompareSpanSummary | null;
  match: CompareMatchKind;
  delta_ms: number | null;
}

export interface CompareTraceHalf {
  trace_id: string;
  name: string;
  status: string;
  start_time: string;
  end_time: string | null;
  agent_name: string | null;
  thread_id: string | null;
  total_cost_usd: number | null;
  total_tokens: number | null;
  span_count: number;
  duration_ms: number | null;
  runner_type: RunnerType;
  runner_name: string | null;
  spans: SpanRow[];
}

export interface CompareSummary {
  duration_delta_ms: number | null;
  tokens_delta: number | null;
  cost_delta_usd: number | null;
  spans_delta: number;
  time_apart_seconds: number | null;
}

export interface CompareTracesResponse {
  trace_a: CompareTraceHalf;
  trace_b: CompareTraceHalf;
  alignment: CompareAlignmentRow[];
  summary: CompareSummary;
}

// ---------------------------------------------------------------------------
// Eval Dataset Editor (Sprint 3)
// ---------------------------------------------------------------------------

export interface DatasetSummary {
  name: string;
  case_count: number;
  modified_at: string;
  created_at: string;
  has_multimodal: boolean;
}

export interface DatasetCaseInputPart {
  type: "text" | "image" | "pdf";
  text?: string;
  path?: string;
  url?: string;
}

export type DatasetCaseInput = string | DatasetCaseInputPart[];

export interface DatasetCase {
  index: number;
  input: DatasetCaseInput;
  expected_output: unknown | null;
  tags: string[];
  metadata: Record<string, unknown>;
  /**
   * Every row field the editor does not model (1.89.0): expected_trajectory,
   * expected_tool_calls, expected_steps, context, source_trace_id... Shown
   * read-only; a save that omits it keeps it.
   */
  extra: Record<string, unknown>;
}

export interface DatasetDetail {
  name: string;
  cases: DatasetCase[];
}

export interface CaseBody {
  input: DatasetCaseInput;
  expected_output?: unknown;
  tags?: string[];
  metadata?: Record<string, unknown>;
  /** Omit to keep the row's other fields on an edit; a dict replaces them. */
  extra?: Record<string, unknown>;
}

export interface DatasetImageUploadResult {
  path: string;
  filename: string;
  size_bytes: number;
}

export interface DatasetImportResult {
  name: string;
  imported: number;
  total: number;
}

export interface DatasetRunEvalResult {
  run_id: string;
  /** Passed / scored cases; null when no case was scored. */
  pass_rate: number | null;
  pass_count: number;
  fail_count: number;
  /** Cases whose agent raised: unscored, neither a pass nor a fail (1.89.0). */
  errored_count?: number;
  total_cases?: number;
  /**
   * "echo" when no agent ran: inputs are echoed back unchanged, so a case
   * passes only where expected_output equals input. A 0% pass rate is normal
   * there and does not mean the eval failed.
   */
  mode?: "echo" | "agent";
  note?: string | null;
  /** "ui" for an agent run, "ui-echo" for a dataset check (1.89.0). */
  kind?: "ui" | "ui-echo";
  /** Case outcomes (1.89.0). */
  cases?: RunEvalCaseCounts;
  /** Scorer verdicts across all cases (1.89.0). */
  checks?: RunEvalCheckCounts;
}

export interface RunEvalCaseCounts {
  passed: number;
  failed: number;
  errored: number;
  pass_rate: number | null;
}

export interface RunEvalCheckCounts {
  passed: number;
  failed: number;
  errored: number;
  skipped: number;
  pass_rate: number | null;
}

// Run eval dialog (1.89.0): GET /datasets/{name}/run-eval/options

export interface RunEvalAgentOption {
  name: string;
  kind: "agent" | "supervisor" | "swarm" | "chain";
  supported: boolean;
  reason: string | null;
}

export type ScorerFamily = "deterministic" | "llm" | "trajectory" | "rag";

/** A row field a scorer reads. */
export type ScorerNeed =
  | "expected_output"
  | "expected_trajectory"
  | "expected_tool_calls"
  | "expected_steps"
  | "context";

export interface ScorerParam {
  name: string;
  type: "string" | "number" | "integer";
  required: boolean;
  default: string | number | null;
  minimum: number | null;
  maximum: number | null;
  description: string;
}

export interface ScorerOption {
  name: string;
  family: ScorerFamily;
  description: string;
  params: ScorerParam[];
  needs: ScorerNeed[];
  /** Calls a model, so it takes a judge model. */
  judge: boolean;
  /** Reads the agent's tool calls: a dataset check cannot feed it. */
  needs_agent: boolean;
  available: boolean;
  unavailable_reason: string | null;
  /** Per needed field, how many rows lack it. */
  rows_missing: Partial<Record<ScorerNeed, number>>;
}

export interface JudgeModelOption {
  provider: string;
  model: string;
}

export interface DatasetFeatures {
  rows: number;
  with_expected_output: number;
  with_expected_trajectory: number;
  with_expected_tool_calls: number;
  with_expected_steps: number;
  with_context: number;
  /** Rows whose input carries an image or a PDF. */
  multimodal: number;
}

export interface RunEvalOptions {
  agents: RunEvalAgentOption[];
  scorers: ScorerOption[];
  /** Only models whose provider has a key on the server. */
  judge_models: JudgeModelOption[];
  dataset_features: DatasetFeatures;
  echo_available: boolean;
}

export type JudgeScale = "binary" | "0-1" | "1-5" | "1-10";

export interface BuiltinScorerSpec {
  type: "builtin";
  name: string;
  params?: Record<string, string | number>;
  /** Only for scorers with `judge: true`; the server default when omitted. */
  model?: JudgeModelOption;
}

export interface LLMJudgeScorerSpec {
  type: "llm_judge";
  name?: string;
  criteria: string;
  scale: JudgeScale;
  threshold: number;
  model?: JudgeModelOption;
}

export interface GEvalScorerSpec {
  type: "geval";
  name?: string;
  criteria: string;
  evaluation_steps?: string[];
  rubric?: [number, string][];
  scale: JudgeScale;
  threshold: number;
  model?: JudgeModelOption;
}

export type ScorerSpec = BuiltinScorerSpec | LLMJudgeScorerSpec | GEvalScorerSpec;

export interface RunEvalBody {
  /** Required unless mode is "echo"; must be absent when it is. */
  agent_name?: string;
  mode?: "agent" | "echo";
  /** A plain string is the built-in of that name. */
  scorers: (string | ScorerSpec)[];
  run_name?: string;
}

// ---------------------------------------------------------------------------
// Filter presets (Sprint 3)
// ---------------------------------------------------------------------------

export interface FilterPreset {
  id: string;
  name: string;
  filters: TraceFilters;
  created_at: string;
}

/** What produced an eval run (1.89.0). `null` = recorded before kinds existed. */
export type EvalRunKind = "evaluate" | "cli" | "pytest" | "optimize" | "ui" | "ui-echo";

/** The Agent-CI gate's verdict on a run. */
export type EvalGateOutcome = "passed" | "failed" | "invalid";

/** One threshold the gate checked, with the value the run reached. */
export interface EvalGateCheck {
  metric: string;
  field: string;
  minimum: number;
  /** `null` when the metric was not present in the run (the gate fails it). */
  actual: number | null;
  passed: boolean;
}

/** The Agent-CI evidence recorded on a run; every part is optional. */
export interface EvalGate {
  outcome: EvalGateOutcome | string | null;
  thresholds?: Record<string, number> | null;
  checks?: EvalGateCheck[] | null;
  error?: {
    errored?: number;
    scored?: number;
    error_rate?: number;
    max_error_rate?: number | null;
  } | null;
  invalid_reasons?: string[] | null;
  baseline?: {
    run_id?: string;
    run_name?: string;
    pass_rate_delta?: number | null;
    case_pass_rate_delta?: number | null;
    regressed_count?: number;
    errored_count?: number;
    unmatched_count?: number;
    tolerance?: number | null;
    pass_rate_comparable?: boolean;
  } | null;
  git_sha?: string | null;
  git_branch?: string | null;
}

/** `metadata.hardening`: the recommendations `harden()` made for this run. */
export interface EvalHardening {
  created_at?: string | null;
  model?: string | null;
  failure_count?: number | null;
  recommendations?: { target?: string; recommendation?: string; rationale?: string }[];
}

/** `metadata.scorer_config`: how each scorer was configured. Never keys. */
export type EvalScorerConfig = Record<string, { type?: string; [key: string]: unknown }>;

export interface EvalRunMetadata {
  gate_outcome?: EvalGateOutcome | string;
  git_branch?: string;
  git_sha?: string;
  scorer_config?: EvalScorerConfig;
  hardening?: EvalHardening;
  [key: string]: unknown;
}

export interface EvalRunRow {
  run_id: string;
  run_name: string | null;
  dataset_name: string | null;
  agent_name: string | null;
  agent_version: string | null;
  scorers: string[] | null;
  started_at: string | null;
  finished_at: string | null;
  /** Checks (verdicts) that passed. Before 1.89.0 errored/skipped counted as fails. */
  pass_count: number | null;
  fail_count: number | null;
  /**
   * The **checks** rate: passed verdicts / (passed + failed verdicts). What
   * `overall.pass_rate` and `--eval-fail-under` gate. `null` = no verdict applied.
   */
  pass_rate: number | null;
  metadata: EvalRunMetadata | null;
  // 1.89.0
  kind?: EvalRunKind | string | null;
  /** 2 for runs written by 1.89.0+; `null` for runs stored under the old rules. */
  summary_version?: number | null;
  case_count?: number | null;
  case_pass_count?: number | null;
  case_fail_count?: number | null;
  case_error_count?: number | null;
  /** The **cases** rate: passed cases / (passed + failed cases). The headline. */
  case_pass_rate?: number | null;
  check_error_count?: number | null;
  check_skip_count?: number | null;
  /** The agent's spend, from its traces. `null` when no case was traced. */
  agent_cost_usd?: number | null;
  /** False when some case had no trace, so the sum may be partial. */
  agent_cost_known?: boolean;
  /** The judges' spend, from the scoring traces. */
  eval_cost_usd?: number | null;
  eval_cost_known?: boolean;
  /** Alias of the agent cost (0 when unknown). */
  cost_usd?: number;
  avg_latency_ms?: number;
  traced_case_count?: number;
  // Only set on the run-detail response.
  scorer_summary?: Record<
    string,
    { pass: number; fail: number; errored?: number; skipped?: number }
  >;
  gate?: EvalGate | null;
}

/**
 * One scorer's stored verdict on one case. `errored` (the check could not
 * run) and `skipped` (the check does not apply) appear only when set; both
 * carry `passed: false, score: 0` as a placeholder that is never a score.
 */
export interface EvalVerdict {
  passed: boolean;
  score: number;
  reason?: string | null;
  errored?: boolean;
  skipped?: boolean;
  details?: Record<string, unknown> | null;
}

/** A single verdict's outcome. */
export type EvalVerdictOutcome = "passed" | "failed" | "errored" | "skipped";

/** Mirrors `verdict_outcome` (fastaiagent/eval/scorer.py). */
export function evalVerdictOutcome(v: EvalVerdict | null | undefined): EvalVerdictOutcome {
  if (!v) return "failed";
  if (v.errored) return "errored";
  if (v.skipped) return "skipped";
  return v.passed ? "passed" : "failed";
}

export interface EvalCaseRow {
  case_id: string;
  run_id: string;
  ordinal: number;
  input: unknown;
  expected_output: unknown;
  actual_output: unknown;
  trace_id: string | null;
  per_scorer: Record<string, EvalVerdict>;
  /**
   * Infrastructure failure (provider 500, timeout, auth) — the agent never
   * produced an output, so the case was NOT scored and `per_scorer` is empty.
   * Errored is its own outcome: excluded from pass/fail, never a quality miss.
   */
  error?: string | null;
  /** The trace of this case's judge calls (1.89.0). */
  scoring_trace_id?: string | null;
}

/** A case's overall outcome. `errored` is infra, not agent quality. */
export type EvalCaseOutcome = "passed" | "failed" | "errored";

/**
 * Mirrors the backend's `case_outcome_detail` (fastaiagent/eval/compare.py).
 * First match wins:
 *   1. the agent errored (`error` set)            → errored
 *   2. any applicable verdict failed               → failed
 *   3. any verdict errored (check could not run)   → errored
 *   4. no verdict passed (all n/a, or none at all) → errored ("unscored")
 *   5.                                             → passed
 */
export function evalCaseOutcome(c: Pick<EvalCaseRow, "error" | "per_scorer">): EvalCaseOutcome {
  if (c.error) return "errored";
  const per = c.per_scorer && typeof c.per_scorer === "object" ? c.per_scorer : {};
  const outcomes = Object.values(per).map(evalVerdictOutcome);
  if (outcomes.includes("failed")) return "failed";
  if (outcomes.includes("errored")) return "errored";
  if (!outcomes.includes("passed")) return "errored";
  return "passed";
}

export interface EvalRunDetail {
  run: EvalRunRow;
  cases: EvalCaseRow[];
  total_cases: number;
}

export interface EvalCaseFilters {
  scorer?: string | null;
  outcome?: EvalCaseOutcome | null;
  q?: string;
}

/** List filters for `GET /api/evals`. */
export interface EvalRunFilters {
  dataset?: string | null;
  agent?: string | null;
  /** Comma-separated run kinds. */
  kind?: string | null;
  exclude_kind?: string | null;
  q?: string | null;
  /** Comma-separated gate outcomes: passed, failed, invalid, none (never gated). */
  gate?: string | null;
  page?: number;
  page_size?: number;
}

export interface EvalScorerDelta {
  scorer: string;
  passed_before: boolean;
  passed_after: boolean;
  changed: boolean;
  // 1.89.0: `null` when the scorer did not run on that side.
  outcome_before?: EvalVerdictOutcome | null;
  outcome_after?: EvalVerdictOutcome | null;
  /** `null` for an errored or skipped verdict, never its placeholder 0. */
  score_before?: number | null;
  score_after?: number | null;
  score_delta?: number | null;
}

export interface EvalComparePair {
  a: EvalCaseRow;
  b: EvalCaseRow;
  scorer_deltas: EvalScorerDelta[];
}

export interface EvalCompareErroredPair extends EvalComparePair {
  /** Which run's case errored. */
  side: "a" | "b" | "both";
}

export interface EvalCompareResponse {
  run_a: EvalRunRow;
  run_b: EvalRunRow;
  regressed: EvalComparePair[];
  improved: EvalComparePair[];
  unchanged_pass: number;
  unchanged_fail: number;
  /** Checks-rate delta; `null` when either side has no checks rate. Read `pass_rate_comparable`. */
  pass_rate_delta: number | null;
  cost_delta_usd: number;
  // 1.89.0
  errored?: EvalCompareErroredPair[];
  only_in_a?: EvalCaseRow[];
  only_in_b?: EvalCaseRow[];
  /** Pairs whose outcome held while a score moved; lists only the moved scorers. */
  drifted?: EvalComparePair[];
  case_pass_rate_delta?: number | null;
  pass_rate_comparable?: boolean;
  agent_cost_delta_usd?: number | null;
  eval_cost_delta_usd?: number | null;
}

/** A case value's image/PDF stored as a descriptor, its bytes kept out (1.89.0). */
export interface EvalMediaDescriptor {
  type: "image" | "pdf";
  media_type?: string;
  size_bytes?: number;
  sha256?: string;
  media_idx: number;
  source_url?: string | null;
  source_path?: string | null;
  detail?: string | null;
}

/** Raw bytes in a case value, stored as their size only. */
export interface EvalBytesDescriptor {
  type: "bytes";
  size_bytes: number;
}

// --- Optimize runs (optimize_runs / optimize_iterations) ---

/** The winning candidate as persisted (``Candidate.to_dict()``). A ``null``
 *  lever means "unchanged from the agent". */
export interface OptimizeCandidate {
  id?: string;
  parent_id?: string | null;
  origin?: string;
  rationale?: string;
  system_prompt?: string | null;
  fewshot_demos?: { input: unknown; output: unknown }[] | null;
  fact_ids?: number[] | null;
}

export interface OptimizeRunMetadata {
  /** The agent's own system prompt when the run started (1.86.0+). */
  baseline_system_prompt?: string;
  /** Each time the prompt proposer could not run (1.85.0+). */
  proposer_errors?: string[];
  [key: string]: unknown;
}

export interface OptimizeRunRow {
  run_id: string;
  run_name: string | null;
  agent_name: string | null;
  baseline_score: number | null;
  best_score: number | null;
  holdout_baseline_score: number | null;
  holdout_best_score: number | null;
  reverted: number | null;
  stopped_reason: string | null;
  seed: number | null;
  levers: string[] | null;
  config: Record<string, unknown> | null;
  best_candidate: OptimizeCandidate | null;
  baseline_eval_run_id: string | null;
  best_eval_run_id: string | null;
  iteration_count: number | null;
  started_at: string | null;
  finished_at: string | null;
  metadata: OptimizeRunMetadata | null;
}

export interface OptimizeIterationRow {
  iteration_id: string;
  run_id: string;
  ordinal: number;
  iteration: number;
  lever: string;
  candidate_id: string;
  dev_score: number | null;
  accepted: number | null;
  skipped: number | null;
  rationale: string | null;
  eval_run_id: string | null;
}

export interface OptimizeRunsPage {
  rows: OptimizeRunRow[];
  total: number;
  page: number;
  page_size: number;
}

export interface OptimizeRunDetail {
  run: OptimizeRunRow;
  iterations: OptimizeIterationRow[];
  total_iterations: number;
}

// --- Agent simulation (sim_runs / sim_cases) ---

export interface SimulationRunRow {
  run_id: string;
  run_name: string | null;
  agent_name: string | null;
  scenario_count: number | null;
  pass_count: number | null;
  fail_count: number | null;
  pass_rate: number | null;
  started_at: string | null;
  finished_at: string | null;
  metadata: Record<string, unknown> | null;
  /** Scenarios that did not finish or could not be judged (1.89.0). */
  errored_count?: number;
  /** Scenarios that reached a verdict. */
  judged_count?: number;
}

export interface TranscriptTurn {
  turn_index: number;
  role: "user" | "assistant";
  content: string;
  trace_id: string | null;
}

export interface CriterionVerdict {
  criterion: string;
  kind: "success" | "failure";
  passed: boolean;
  reason: string | null;
  /** The judge could not run on this criterion (1.89.0). */
  errored?: boolean;
}

export interface SimulationCase {
  case_id: string;
  run_id: string;
  ordinal: number;
  scenario_name: string | null;
  passed: number; // 0 | 1
  criteria: { success: string[]; failure: string[] };
  per_criterion: CriterionVerdict[];
  transcript: TranscriptTurn[];
  trace_id: string | null;
  /** Why the scenario did not finish or could not be judged (1.89.0). */
  error?: string | null;
}

export interface SimulationRunDetail {
  run: SimulationRunRow;
  cases: SimulationCase[];
  total_cases: number;
}

export interface SimulationRunsPage {
  rows: SimulationRunRow[];
  total: number;
  page: number;
  page_size: number;
}

export interface SimulationCaseFilters {
  outcome?: EvalCaseOutcome | null;
}

export interface PromptListItem {
  name: string;
  latest_version: number | string;
  versions: number;
  linked_trace_count: number;
  registry_is_local: boolean;
}

export interface PromptListResponse {
  rows: PromptListItem[];
  registry_is_local: boolean;
}

/** One prompt version, as every prompt endpoint returns it (1.89.0). */
export interface PromptVersionDetail {
  slug: string;
  version: number;
  /** Fragments resolved: what a run sends. */
  template: string;
  /** As stored, `{{@fragment}}` references intact: what the editor edits. */
  template_raw: string;
  fragments: string[];
  variables: string[];
  metadata: Record<string, unknown>;
}

export interface PromptDetail extends PromptVersionDetail {
  latest_version: number;
  registry_is_local: boolean;
}

export interface PromptVersionRow extends Omit<PromptVersionDetail, "version"> {
  /** The list endpoint has always returned the version as a string. */
  version: string;
  created_at: string | null;
  created_by: string | null;
}

export interface PromptLineage {
  trace_ids: string[];
  eval_run_ids: string[];
}

export interface EvalRunsPage {
  rows: EvalRunRow[];
  total: number;
  page: number;
  page_size: number;
}

export interface EvalTrendPoint {
  started_at: string;
  run_id?: string;
  run_name?: string | null;
  kind?: string | null;
  dataset_name: string | null;
  /** The line this point belongs to: the dataset, or the run name for pytest. */
  series?: string | null;
  /** Cases rate; `null` when no case reached a verdict. */
  case_pass_rate?: number | null;
  /** Checks rate. */
  pass_rate: number | null;
  gate_outcome?: string | null;
}

export interface AnalyticsPoint {
  bucket: string;
  trace_count: number;
  error_count: number;
  error_rate: number;
  cost_usd: number;
  p50_ms: number | null;
  p95_ms: number | null;
  p99_ms: number | null;
}

/**
 * Traces with a non-production source (replay, Playground, learn, eval…),
 * reported apart from production so excluding them never hides spend
 * (1.89.0).
 */
export interface ExperimentsSummary {
  trace_count: number;
  cost_usd: number;
  by_source: Record<string, { trace_count: number; cost_usd: number }>;
}

export interface AnalyticsSummary {
  trace_count: number;
  error_count: number;
  error_rate: number;
  total_cost_usd: number;
  p50_ms: number | null;
  p95_ms: number | null;
  p99_ms: number | null;
  // 1.89.0
  experiments?: ExperimentsSummary;
  experiments_included?: boolean;
}

export interface AnalyticsAgent {
  agent_name: string;
  run_count: number;
  avg_latency_ms?: number;
  total_cost_usd: number;
  avg_cost_usd?: number;
  error_count: number;
}

export interface AnalyticsPayload {
  window_hours: number;
  granularity: "hour" | "day";
  summary: AnalyticsSummary;
  points: AnalyticsPoint[];
  top_slowest_agents: AnalyticsAgent[];
  top_priciest_agents: AnalyticsAgent[];
}

export interface TraceScores {
  trace_id: string;
  guardrail_events: {
    event_id: string;
    guardrail_name: string;
    guardrail_type: string | null;
    position: string | null;
    outcome: string | null;
    score: number | null;
    message: string | null;
    agent_name: string | null;
    timestamp: string | null;
    /** 1.89.0 — shown as a suffix when the endpoint reports it. */
    effect?: GuardrailEffect | null;
  }[];
  eval_cases: {
    case_id: string;
    run_id: string;
    ordinal: number;
    per_scorer: Record<string, EvalVerdict>;
    run_name: string | null;
    dataset_name: string | null;
    started_at: string | null;
    input: unknown;
    expected_output: unknown;
    actual_output: unknown;
  }[];
  /** 1.89.0: this trace IS the judges' scoring of these eval cases. */
  eval_scoring?: {
    case_id: string;
    run_id: string;
    ordinal: number;
    per_scorer: Record<string, EvalVerdict>;
    /** The agent trace of the case that was scored. */
    trace_id: string | null;
    run_name: string | null;
    dataset_name: string | null;
    started_at: string | null;
  }[];
  /** 1.89.0: this trace is a simulated conversation. */
  simulation_cases?: {
    case_id: string;
    run_id: string;
    ordinal: number;
    scenario_name: string | null;
    passed: number | boolean;
    per_criterion: CriterionVerdict[];
    error: string | null;
    run_name: string | null;
    started_at: string | null;
  }[];
}

export interface ThreadTrace {
  trace_id: string;
  name: string;
  start_time: string;
  end_time: string | null;
  status: string;
  span_count: number;
  duration_ms: number | null;
  agent_name: string | null;
  thread_id: string | null;
  total_cost_usd: number | null;
  total_tokens: number | null;
  runner_type: RunnerType;
  runner_name: string | null;
}

export interface ThreadDetail {
  thread_id: string;
  traces: ThreadTrace[];
}

export interface GuardrailEvent {
  event_id: string;
  trace_id: string | null;
  span_id: string | null;
  guardrail_name: string;
  guardrail_type: string | null;
  position: string | null;
  outcome: string | null;
  score: number | null;
  message: string | null;
  agent_name: string | null;
  timestamp: string | null;
  metadata: Record<string, unknown>;
  false_positive: boolean;
  false_positive_at: string | null;
  // 1.57.0 — what the failure cost. `action` is what the rule was configured
  // to do, `action_taken` what it actually did; they differ whenever the
  // action could not get what it asked for (an errored check always blocks,
  // a mask with nothing to mask blocks). Null for events recorded earlier.
  action: string | null;
  action_taken: string | null;
  severity: string | null;
  floor: boolean;
  // 1.89.0 (local.db v25). All null on rows written before 1.89.0 ("legacy").
  /** What the check did to the run, not just its verdict. */
  effect: GuardrailEffect | null;
  /** Why a rewrite or re-ask ended as a block (the caller refused it). */
  final_reason: string | null;
  latency_ms: number | null;
  attempt: number | null;
  /** Rows of one position pass share a batch. */
  batch_id: string | null;
  guardrail_span_id: string | null;
  judged_ref: string | null;
  /** The rule as it ran. */
  rule: GuardrailRuleSnapshot | null;
  blocking: boolean | null;
  on_error: string | null;
  origin: "local" | "plane" | string | null;
}

/**
 * What a check did to the run (1.89.0):
 * none — clean pass; warned — failed, recorded, run continued unchanged;
 * rewritten — payload masked/overridden and used; reasked — model asked again;
 * blocked — run stopped here; observed — observe-only rule, nothing enforced;
 * unchecked — could not run, ``on_error=allow`` let it through.
 */
export type GuardrailEffect =
  | "none"
  | "warned"
  | "rewritten"
  | "reasked"
  | "blocked"
  | "observed"
  | "unchecked";

export interface GuardrailRuleSnapshot {
  name?: string;
  guardrail_type?: string;
  position?: string;
  /** ``llm`` is reduced to provider/model. */
  config?: Record<string, unknown>;
  blocking?: boolean;
  description?: string | null;
  on_error?: string | null;
  action?: string | null;
  severity?: string | null;
  floor?: boolean;
  origin?: string | null;
}

export interface GuardrailEventsPage {
  rows: GuardrailEvent[];
  total: number;
  page: number;
  page_size: number;
  /**
   * Only when ``total`` is 0: guardrail spans recorded before 1.89.0, when
   * checks were not yet written as rows by default.
   */
  legacy_guardrail_spans?: number;
}

// Sprint 2 — Guardrail Event Detail
export interface GuardrailTrigger {
  kind: "agent_input" | "agent_output" | "tool_call" | "tool_result" | "unknown";
  text: string | null;
  content_type: string;
  span_name?: string;
  status?: string;
}

export interface GuardrailContextSpan {
  span_id: string;
  name: string;
  start_time: string | null;
  end_time: string | null;
  status: string | null;
  input: string | null;
  output: string | null;
}

export interface GuardrailEventDetail {
  event: GuardrailEvent;
  /**
   * 1.89.0 — exactly the text the rule judged. ``recorded`` is false for a
   * row written before 1.89.0 or one whose payload was purged.
   */
  judged: { text: string | null; recorded: boolean };
  /** Compatibility: read from the parent span, which may hold rewritten text. */
  trigger: GuardrailTrigger;
  context: {
    spans: GuardrailContextSpan[];
    /** Since 1.89.0, the other rows of the same position pass (batch). */
    sibling_events: GuardrailEvent[];
  };
}

export interface FalsePositiveResponse {
  event_id: string;
  false_positive: boolean;
  /** Null once the flag is cleared (1.89.0). */
  false_positive_at: string | null;
}

export interface AgentSummary {
  agent_name: string;
  run_count: number;
  success_rate: number;
  error_count: number;
  avg_latency_ms: number;
  avg_cost_usd: number;
  last_run: string;
  workflows?: { runner_type: "chain" | "swarm" | "supervisor"; name: string }[];
}

export interface ForkModifications {
  prompt?: string;
  input?: Record<string, unknown> | Array<Record<string, unknown>>;
  /**
   * v1.14.1+: map of tool name → canned response. Each entry installs
   * a stub FunctionTool via ForkedReplay.with_tool_override on the
   * backend so the rerun's LLM receives the canned value in place of
   * calling the original tool.
   */
  tool_overrides?: Record<string, unknown>;
  /** 1.89.0: how the rerun calls the LLM. */
  determinism?: ReplayDeterminism;
  /** 1.89.0: ``recorded`` only — what to do when the rerun makes more LLM calls than recorded. */
  on_miss?: ReplayOnMiss;
  /** 1.89.0: what the rerun does with tools you did not stub or allow. */
  tool_policy?: ReplayToolPolicy;
  /** 1.89.0: tools that run live under any policy. */
  allow_tools?: string[];
  /**
   * @deprecated since v1.14.1 — was a no-op for agent reruns
   * (silently routed to modify_state which the agent path ignored).
   * Use ``tool_overrides`` instead.
   */
  tool_response?: Record<string, unknown>;
  config?: Record<string, unknown>;
  state?: Record<string, unknown>;
}

export interface TraceFilters {
  agent?: string | null;
  status?: string | null;
  q?: string;
  thread_id?: string | null;
  runner_type?: RunnerType | null;
  runner_name?: string | null;
  framework?: string | null;
  /** 1.89.0: hide replay, Playground, learn and eval traces. */
  exclude_experiments?: boolean;
  since?: string;
  until?: string;
  min_duration_ms?: number;
  max_duration_ms?: number;
  min_cost?: number;
  max_cost?: number;
  min_tokens?: number;
  page?: number;
  page_size?: number;
}

export interface WorkflowSummary {
  runner_type: WorkflowRunnerType;
  workflow_name: string;
  run_count: number;
  success_rate: number;
  error_count: number;
  avg_latency_ms: number;
  avg_cost_usd: number;
  last_run: string;
  node_count: number | null;
}

export interface WorkflowListResponse {
  workflows: WorkflowSummary[];
  registered?: boolean;
}

// ---------------------------------------------------------------------------
// Workflow topology (used by the Local UI's React Flow canvas)
// ---------------------------------------------------------------------------

export type TopologyNodeType =
  | "agent"
  | "tool"
  | "condition"
  | "parallel"
  | "hitl"
  | "start"
  | "end"
  | "transformer"
  | "supervisor";

export interface TopologyNode {
  id: string;
  type: TopologyNodeType;
  label: string;
  agent_name?: string;
  tool_name?: string;
  model?: string;
  provider?: string;
  description?: string;
  tool_count?: number;
}

export type TopologyEdgeType =
  | "sequential"
  | "conditional"
  | "handoff"
  | "delegation";

export interface TopologyEdge {
  from: string;
  to: string;
  type: TopologyEdgeType;
  label?: string;
  condition?: string;
  is_cyclic?: boolean;
}

export interface TopologyTool {
  owner: string;
  name: string;
  type: string;
}

export interface WorkflowTopology {
  name: string;
  type: "chain" | "swarm" | "supervisor";
  nodes: TopologyNode[];
  edges: TopologyEdge[];
  entrypoint: string | null;
  tools: TopologyTool[];
  knowledge_bases: { owner: string; name: string }[];
  max_handoffs?: number;
  max_delegation_rounds?: number;
}

// ---------------------------------------------------------------------------
// Cost breakdown
// ---------------------------------------------------------------------------

export type CostGroupBy = "model" | "agent" | "node";

export interface CostByModelRow {
  model: string;
  calls: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
}

export interface CostByAgentRow {
  agent: string;
  runs: number;
  avg_tokens: number;
  avg_cost_usd: number;
  total_cost_usd: number;
}

export interface CostByNodeRow {
  node: string;
  executions: number;
  avg_duration_ms: number;
  avg_cost_usd: number;
  percent_of_total: number;
}

export interface CostBreakdownResponse {
  group_by: CostGroupBy;
  period: string;
  chain_name?: string | null;
  rows: CostByModelRow[] | CostByAgentRow[] | CostByNodeRow[];
  /** 1.89.0: the experiment traces in this period, whether or not they are in ``rows``. */
  experiments?: ExperimentsSummary;
}

export type ToolOrigin =
  | "function"
  | "mcp"
  | "rest"
  | "kb"
  | "custom"
  | "unknown";

export interface RegisteredTool {
  name: string;
  description: string;
  origin: ToolOrigin;
  used: boolean;
}

export interface UsedTool {
  name: string;
  origin: ToolOrigin;
  call_count: number;
  error_count: number;
  success_rate: number;
  avg_latency_ms: number;
  last_used: string;
  registered: boolean;
}

export interface AgentToolsResponse {
  agent_name: string;
  registered: RegisteredTool[];
  used: UsedTool[];
}

export interface KbSummary {
  name: string;
  path: string;
  chunk_count: number;
  doc_count: number;
  last_updated: string;
  size_bytes: number;
}

export interface KbListResponse {
  root: string;
  collections: KbSummary[];
}

export interface KbDetail {
  name: string;
  path: string;
  chunk_count: number;
  doc_count: number;
  size_bytes: number;
  last_updated: string;
  metadata_keys: string[];
}

export interface KbDocumentRow {
  source: string;
  chunk_count: number;
  preview: string;
  metadata: Record<string, unknown>;
}

export interface KbDocumentsResponse {
  total: number;
  page: number;
  page_size: number;
  documents: KbDocumentRow[];
}

export interface KbChunk {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
  index: number;
  start_char: number;
  end_char: number;
}

export interface KbChunksResponse {
  source: string;
  chunks: KbChunk[];
}

export interface KbSearchHit {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
  score: number;
  source: string | null;
  index: number;
}

export interface KbSearchResponse {
  query: string;
  top_k: number;
  search_type: string;
  results: KbSearchHit[];
}

export interface KbLineageAgent {
  agent_name: string;
  retrieval_count: number;
}

export interface KbLineageTrace {
  trace_id: string;
  name: string;
  start_time: string;
  status: string;
  agent_name: string | null;
}

export interface KbLineageResponse {
  kb_name: string;
  retrieval_count: number;
  agents: KbLineageAgent[];
  recent_traces: KbLineageTrace[];
}

// ---------------------------------------------------------------------------
// Prompt Playground
// ---------------------------------------------------------------------------

export interface PlaygroundProviderInfo {
  provider: string;
  models: string[];
  has_key: boolean;
  env_var: string | null;
}

export interface PlaygroundModelsResponse {
  providers: PlaygroundProviderInfo[];
}

/**
 * `null` means "don't send this parameter" — not "send the default".
 *
 * Anthropic rejects `temperature` and `top_p` together, and Claude 5 rejects
 * `top_p` outright, so shipping 1.0 defaults broke every Anthropic call. Only
 * forward what the user explicitly enabled.
 */
export interface PlaygroundParameters {
  temperature: number | null;
  max_tokens: number;
  top_p: number | null;
}

export interface PlaygroundRunRequest {
  provider: string;
  model: string;
  prompt_template: string;
  variables: Record<string, string>;
  system_prompt?: string;
  parameters: PlaygroundParameters;
  image_b64?: string;
  image_media_type?: string;
  /** Point at an AI gateway / proxy / self-hosted server. http(s) only. */
  base_url?: string;
  /** Per-request credential. Never persisted client- or server-side. */
  api_key?: string;
}

export interface PlaygroundRunResponse {
  response: string;
  model: string;
  provider: string;
  latency_ms: number;
  tokens: { input: number; output: number };
  cost_usd: number | null;
  trace_id: string | null;
  /**
   * 1.89.0: false when the trace went to a different store than this UI
   * reads, so a link to it would 404. Absent (older server) = true.
   */
  trace_in_ui_db?: boolean;
  finish_reason: string | null;
}

export interface PlaygroundDoneMetadata {
  model: string;
  provider: string;
  latency_ms: number;
  tokens: { input: number; output: number };
  cost_usd: number | null;
  trace_id: string | null;
  /** See {@link PlaygroundRunResponse.trace_in_ui_db}. */
  trace_in_ui_db?: boolean;
}

export type PlaygroundStreamEvent =
  | { event: "token"; text: string }
  | { event: "done"; metadata: PlaygroundDoneMetadata }
  | {
      event: "error";
      /** Always a string (1.89.0): a 422's list `detail` crashed the page. */
      message: string;
      correlation_id?: string;
      field_errors?: { field: string; message: string }[];
      kind?: string;
      provider_status?: number;
      provider_message?: string;
      hint?: string;
    };

export interface SaveAsEvalRequest {
  dataset_name: string;
  input: unknown;
  expected_output: unknown;
  system_prompt?: string;
  model?: string;
  provider?: string;
}

export interface SaveAsEvalResponse {
  dataset_name: string;
  path: string;
  line_count: number;
}

// ---------------------------------------------------------------------------
// Agent Dependency Graph
// ---------------------------------------------------------------------------

export interface AgentDepNode {
  name: string;
  type: "agent" | "supervisor" | "worker";
  model: string | null;
  provider: string | null;
}

export interface AgentDepTool {
  name: string;
  origin: string;
  registered: boolean;
  calls: number;
  success_rate: number;
  avg_latency_ms: number;
}

export interface AgentDepKB {
  name: string;
  backend: string;
  documents: number | null;
  chunks: number | null;
  unresolved?: boolean;
}

export interface AgentDepPrompt {
  name: string;
  version: string | null;
  variables: string[];
  preview?: string;
}

export interface AgentDepGuardrail {
  name: string | null;
  guardrail_type: string | null;
  position: string | null;
  // 1.89.0 — absent on payloads from older servers.
  action?: string | null;
  blocking?: boolean | null;
  on_error?: string | null;
  severity?: string | null;
  floor?: boolean | null;
}

/**
 * 1.89.0 — a rule that ran for the agent (it has event rows) but that the
 * agent does not declare, e.g. one distributed by a control plane.
 */
export interface AgentDepObservedGuardrail {
  name: string;
  position: string | null;
  guardrail_type: string | null;
  last_seen: string | null;
  checks: number;
  origin: string | null;
}

export interface AgentDepHandoff {
  from: string;
  to: string;
}

export interface AgentDependencies {
  agent: AgentDepNode;
  tools: AgentDepTool[];
  knowledge_bases: AgentDepKB[];
  prompts: AgentDepPrompt[];
  guardrails: AgentDepGuardrail[];
  observed_guardrails?: AgentDepObservedGuardrail[];
  model: { provider: string | null; model: string | null };
  sub_agents?: AgentDependencies[];
  peers?: AgentDepNode[];
  handoffs?: AgentDepHandoff[];
  parent?: { name: string | null; type: "supervisor" | "swarm" };
  role?: string;
  unresolved?: boolean;
}
