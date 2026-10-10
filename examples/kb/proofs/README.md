# Proofs behind "Retrieval Breaks at the Boundaries"

The six claims on [Retrieval Breaks at the Boundaries](../../../docs/knowledge-base/kb-boundaries.md),
each as a script you can run against the published SDK. Every script works in a
scratch folder of its own, so nothing touches your project's knowledge bases.

| Section of the page | Script | Needs | What it shows |
|---|---|---|---|
| 1 · Between a document and its chunks | `proof_1_chunks.py` | nothing | the seams, the offsets, and that `chunk_overlap` carries nothing across a seam |
| 2 · Between text and vector | `proof_2_embedder.py` | `fastaiagent[kb]` | the same query on two embedders; a different dimension is refused on reopen; a reopen embeds one probe word |
| 3 · Between meaning and words | `proof_3_matchers.py` | `fastaiagent[kb]` | vector, keyword and hybrid scores on three queries; the empty-side short-circuit; `0.7 × 1.0 + 0.3 × 1.0` |
| 4 · Between the knowledge base and the model | `proof_4_tool.py` | `fastaiagent[kb]`, `OPENAI_API_KEY` | the model writes the query; two spans; what the export policy drops |
| 5 · Between one process and the next | `proof_5_restart.py` | `fastaiagent[kb]` | three processes, one `kb.sqlite`; indexes rebuilt, nothing re-embedded; a delete that persists |
| 6 · Between a laptop and a fleet | `proof_6_backends.py` | `fastaiagent[kb,chroma,qdrant]` | the cosine contract on FAISS, Chroma and Qdrant; the Chroma float32 defect in 1.87.0 |

```sh
cd examples/kb/proofs
python proof_1_chunks.py && python proof_2_embedder.py && python proof_3_matchers.py \
  && python proof_5_restart.py && python proof_6_backends.py
zsh -lc 'python proof_4_tool.py'          # the live one
```

Chroma runs in-process and Qdrant in-memory: no server is needed. The local
FastEmbed model (`BAAI/bge-small-en-v1.5`) downloads on first use.

`../tests/test_proofs.py` runs the five offline scripts in CI, so the outputs the page
quotes can't drift from what the SDK does.
