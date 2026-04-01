"""Benchmark: LocalKB search. Target: search <500ms over 1000 chunks."""

import time

from fastaiagent.kb import LocalKB
from fastaiagent.kb.embedding import SimpleEmbedder


def bench_kb_search(num_chunks=1000):
    kb = LocalKB(name="bench", embedder=SimpleEmbedder(dimensions=128))

    # Add chunks
    for i in range(num_chunks):
        kb.add(
            f"Document chunk {i}: This is sample content about topic {i % 50}. "
            f"It contains information relevant to category {i % 10}."
        )

    # Benchmark search
    start = time.monotonic()
    results = kb.search("topic 25 category 5", top_k=10)
    elapsed = time.monotonic() - start

    return elapsed, len(results)


if __name__ == "__main__":
    n = 1000
    elapsed, result_count = bench_kb_search(n)
    elapsed_ms = elapsed * 1000
    print(f"Chunks: {n}")
    print(f"Results: {result_count}")
    print(f"Search time: {elapsed_ms:.1f}ms")
    print(f"Target: <500ms — {'PASS' if elapsed_ms < 500 else 'FAIL'}")
