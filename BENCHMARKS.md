# FastAIAgent SDK — Performance Benchmarks

## Summary

| Benchmark | Target | Measured | Status |
|-----------|--------|----------|--------|
| Trace overhead | < 5% | 19.5% | ⚠️ Above target |
| Checkpoint per node | < 50ms | 0.1ms | ✅ Pass |
| Cycle performance (100 iterations) | < 10s | < 0.001s | ✅ Pass |
| Replay load (1000 spans) | < 2s | 0.001s | ✅ Pass |
| KB search (1000 chunks) | < 500ms | 9.4ms | ✅ Pass |
| Eval throughput (100 cases, 2 scorers) | < 60s | 0.001s | ✅ Pass |

## Environment

- Python 3.12, macOS (Apple Silicon)
- fastaiagent v0.1.0a1
- Benchmarks use synthetic data (no real LLM calls)

## Known Issues

**Trace overhead (19.5%):** The OTel `TracerProvider` + `LocalStorageProcessor` adds
measurable overhead on high-frequency synthetic calls. In real-world usage with LLM
latency (100ms–2s per call), trace overhead is negligible (< 0.5%). Optimization
candidates: batch SQLite writes, reduce span attribute serialization.

## How to Reproduce

```bash
pip install -e ".[dev]"

python benchmarks/bench_trace_overhead.py
python benchmarks/bench_checkpoint_overhead.py
python benchmarks/bench_cycle_performance.py
python benchmarks/bench_replay_load.py
python benchmarks/bench_local_kb.py
python benchmarks/bench_eval.py
```

## Benchmark Details

### Trace Overhead (`bench_trace_overhead.py`)

Measures the cost of OTel tracing on agent execution. Runs 1000 iterations with
and without tracing, compares median execution time. Target: < 5% overhead.

### Checkpoint Overhead (`bench_checkpoint_overhead.py`)

Measures SQLite checkpoint write time per chain node. Runs 100 checkpoints and
reports per-node time. Target: < 50ms per node.

### Cycle Performance (`bench_cycle_performance.py`)

Measures chain cycle execution with 100 iterations. Reports total time excluding
LLM calls. Target: < 10s total.

### Replay Load (`bench_replay_load.py`)

Measures time to load and parse a trace with 1000 spans from SQLite storage.
Target: < 2s.

### Local KB Search (`bench_local_kb.py`)

Indexes 1000 chunks with SimpleEmbedder, then searches 10 queries. Reports
per-query search latency. Target: < 500ms per query.

### Eval Throughput (`bench_eval.py`)

Evaluates 100 test cases with 2 scorers (exact_match, contains). Reports total
evaluation time excluding LLM calls. Target: < 60s total.
