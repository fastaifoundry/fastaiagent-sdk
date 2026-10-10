# Proofs behind "The Model Call Breaks at the Boundaries"

The six claims on [The Model Call Breaks at the Boundaries](../../../docs/llm/llm-boundaries.md),
each as a script you can run against the published SDK.

Three run offline. The wire-shape ones point `LLMClient` at `_recorder.py`, a local
endpoint that records the request and answers with a canned reply, so the bodies
they print are the SDK's own bytes and only the model is canned. Three call real
providers; a missing key skips that provider with a line saying so.

| Section of the page | Script | Needs | What it shows |
|---|---|---|---|
| 1 · Between your messages and the wire | `proof_1_wire.py` | nothing | one conversation rendered for `openai`, `custom` and `anthropic` |
| 2 · Between the reply and your code | `proof_2_parse.py` | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY` | four providers, one `LLMResponse` shape, for a plain reply and a tool call |
| 3 · Between a provider's name and the code | `proof_3_presets.py` | nothing | the preset table; a gateway preset and what its flags do to the body |
| 4 · Between one reply and a stream of it | `proof_4_stream.py` | `OPENAI_API_KEY` | `complete()` vs `stream()`, the event sequence, the span, and an agent's stream |
| 5 · Between free text and a typed object | `proof_5_structured.py` | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | `output_type` on two providers; the agent's re-ask on a bad reply |
| 6 · Between the call and the numbers | `proof_6_numbers.py` | nothing | cost by prefix; retries on 429/503; the two serializers; the export policy |

```sh
cd examples/llm/proofs
python proof_1_wire.py && python proof_3_presets.py && python proof_6_numbers.py
zsh -lc 'python proof_2_parse.py && python proof_4_stream.py && python proof_5_structured.py'   # the live ones
```

`../tests/test_proofs.py` runs the three offline scripts in CI, so the outputs the page
quotes can't drift from what the SDK does.
