/**
 * Shared TypeScript types mirroring the FastAPI server's Pydantic models.
 */

export type RunnerType = "agent" | "chain" | "swarm" | "supervisor";

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

export interface RerunResult {
  fork_id: string;
  new_trace_id: string | null;
  new_output: unknown;
  original_output: unknown;
  steps_executed: number;
}

export interface ComparisonResult {
  original_steps: ReplayStep[];
  new_steps: ReplayStep[];
  diverged_at: number | null;
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
  pass_rate: number | null;
  pass_count: number;
  fail_count: number;
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

export interface EvalRunRow {
  run_id: string;
  run_name: string | null;
  dataset_name: string | null;
  agent_name: string | null;
  agent_version: string | null;
  scorers: string[] | null;
  started_at: string | null;
  finished_at: string | null;
  pass_count: number | null;
  fail_count: number | null;
  pass_rate: number | null;
  metadata: Record<string, unknown> | null;
  // Added in 0.9.3 — aggregated from eval_cases.trace_id → spans.
  cost_usd?: number;
  avg_latency_ms?: number;
  case_count?: number;
  // Only set on the run-detail response.
  scorer_summary?: Record<string, { pass: number; fail: number }>;
}

export interface EvalCaseRow {
  case_id: string;
  run_id: string;
  ordinal: number;
  input: unknown;
  expected_output: unknown;
  actual_output: unknown;
  trace_id: string | null;
  per_scorer: Record<string, { passed: boolean; score: number; reason?: string | null }>;
}

export interface EvalRunDetail {
  run: EvalRunRow;
  cases: EvalCaseRow[];
  total_cases: number;
}

export interface EvalCaseFilters {
  scorer?: string | null;
  outcome?: "passed" | "failed" | null;
  q?: string;
}

export interface EvalScorerDelta {
  scorer: string;
  passed_before: boolean;
  passed_after: boolean;
  changed: boolean;
}

export interface EvalComparePair {
  a: EvalCaseRow;
  b: EvalCaseRow;
  scorer_deltas: EvalScorerDelta[];
}

export interface EvalCompareResponse {
  run_a: EvalRunRow;
  run_b: EvalRunRow;
  regressed: EvalComparePair[];
  improved: EvalComparePair[];
  unchanged_pass: number;
  unchanged_fail: number;
  pass_rate_delta: number;
  cost_delta_usd: number;
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

export interface PromptDetail {
  slug: string;
  latest_version: number;
  template: string;
  variables: string[];
  metadata: Record<string, unknown>;
  registry_is_local: boolean;
}

export interface PromptVersionRow {
  slug: string;
  version: string;
  template: string | null;
  variables: string | null;
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
  pass_rate: number;
  dataset_name: string | null;
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

export interface AnalyticsSummary {
  trace_count: number;
  error_count: number;
  error_rate: number;
  total_cost_usd: number;
  p50_ms: number | null;
  p95_ms: number | null;
  p99_ms: number | null;
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
  }[];
  eval_cases: {
    case_id: string;
    run_id: string;
    ordinal: number;
    per_scorer: Record<string, { passed: boolean; score: number; reason?: string | null }>;
    run_name: string | null;
    dataset_name: string | null;
    started_at: string | null;
    input: unknown;
    expected_output: unknown;
    actual_output: unknown;
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
}

export interface GuardrailEventsPage {
  rows: GuardrailEvent[];
  total: number;
  page: number;
  page_size: number;
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
  trigger: GuardrailTrigger;
  context: {
    spans: GuardrailContextSpan[];
    sibling_events: GuardrailEvent[];
  };
}

export interface FalsePositiveResponse {
  event_id: string;
  false_positive: boolean;
  false_positive_at: string;
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
  input?: Record<string, unknown>;
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
  runner_type: Exclude<RunnerType, "agent">;
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

export interface PlaygroundParameters {
  temperature: number;
  max_tokens: number;
  top_p: number;
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
}

export interface PlaygroundRunResponse {
  response: string;
  model: string;
  provider: string;
  latency_ms: number;
  tokens: { input: number; output: number };
  cost_usd: number | null;
  trace_id: string | null;
  finish_reason: string | null;
}

export interface PlaygroundDoneMetadata {
  model: string;
  provider: string;
  latency_ms: number;
  tokens: { input: number; output: number };
  cost_usd: number | null;
  trace_id: string | null;
}

export type PlaygroundStreamEvent =
  | { event: "token"; text: string }
  | { event: "done"; metadata: PlaygroundDoneMetadata }
  | { event: "error"; message: string };

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
  model: { provider: string | null; model: string | null };
  sub_agents?: AgentDependencies[];
  peers?: AgentDepNode[];
  handoffs?: AgentDepHandoff[];
  parent?: { name: string | null; type: "supervisor" | "swarm" };
  role?: string;
  unresolved?: boolean;
}
