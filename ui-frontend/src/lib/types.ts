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
}

export interface GuardrailEventsPage {
  rows: GuardrailEvent[];
  total: number;
  page: number;
  page_size: number;
}

export interface AgentSummary {
  agent_name: string;
  run_count: number;
  success_rate: number;
  error_count: number;
  avg_latency_ms: number;
  avg_cost_usd: number;
  last_run: string;
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
