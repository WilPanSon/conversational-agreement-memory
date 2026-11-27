# Common-ground memory compression: development pilot (v0.1)

## Research question
Does incrementally compressed LLM memory preserve *the currently mutually established meaning* of a referential convention, as distinct from just the fact that a referent was suggested? The pilot is a first test of the need for grounding-aware memory; no model has been evaluated yet.

## Contents

- `pilot_development.jsonl`: 320 complete labeled dialogues from 80 scenario families (4 variants each). **Private labels present.** Each row contains 24 speaker-attributed turns, four 6-turn update blocks, three questions with gold answers, and evidence-turn indices.
- `pilot_review.csv`: compact spreadsheet-compatible view for manual inspection; **contains gold labels**.
- `human_annotation.csv`: separate annotation sheet with blank label fields for reviewers (contains opaque IDs and transcripts with blank answer fields). Rows are ordered by opaque ID, with family and condition columns omitted.
- `full_history_prompts.jsonl`: 960 model-facing full-history questions, **without gold labels or condition metadata**. Send only the `prompt` field to the model and retain the opaque ID for scoring.
- `answer_key.jsonl`: 960 private answers for scoring; **never show to a model**.
- `generate_pilot.py`: deterministic, standard-library-only dataset generator.
- `make_eval_inputs.py`: creates prompt and key files from the generator output.
- `score_predictions.py`: preliminary exact-match scorer for predictions JSONL.
- `manifest.json`: provenance, counts, seed and validation status.
- `pilot_first_family.jsonl`: the first labeled family in machine-readable form.
- `preview_examples.txt`: one complete family (all four conditions) for inspection.

## Experimental conditions

`accepted`: initial mapping mutually accepted, later replacement proposed but not accepted; initial referent remains active.

`rejected`: initial mapping explicitly rejected, later alternative proposed but not accepted; no active referent.

`unresolved`: initial mapping explicitly postponed, later alternative proposed but not accepted; no active referent.

`revised`: initial mapping mutually accepted, later replacement explicitly accepted; newer referent is active and original mapping is superseded.

Candidate order, proposing speaker, and initial proposal position are sampled per family and shared across its variants. The final block avoids restating the original referent, and the grounding-status question identifies the proposal by its label.

All dialogue variants in a family share the same original entity–label proposal and the same factual-recall answer. The accepted/rejected/unresolved transcripts are identical except for *one reply*; revised differs from accepted in *one later reply*. There are several unrelated agreed names as distractors.

## Running / regenerating

```bash
python generate_pilot.py --families 80 --seed 20261119
python make_eval_inputs.py
```

To regenerate, run both commands in the directory containing these scripts. **Do not generate the held-out test split from templates you are still tuning.** Development examples can be used for pilot-driven refinement, but final validation and test scenarios need separately reviewed templates, paraphrases, and family-level partitioning.

## Proposed first evaluation

1. Human-annotate at least 20% of development dialogues *before* treating template labels as reliable.
2. Run Qwen2.5-7B-Instruct on `full_history_prompts.jsonl`, with deterministic decoding. The model returns one of the five candidate entity IDs, `UNRESOLVED`, or the requested status. Use a frozen checkpoint and record the prompt, tokenizer, and decoding settings.
3. Check full-history common-ground accuracy vs factual recall. If there is a severe model-inference deficit even with full history, memory compression alone will not resolve it.
4. Use exactly the same four 6-turn blocks to generate recursive summary, grounding-aware prose, and structured grounding-state memory. **The memory builder must never receive the future query.** Keep 128- and 256-token budgets for the pilot, using the Qwen tokenizer to count all memory tokens.
5. Ask the same three questions from **memory only**, with matching model and answer settings; compare reference accuracy, false-agreement rate, revision accuracy, and factual recall. Count memory construction cost and total tokens.
6. Treat BM25 retrieval separately because it can store an external index; it is not storage-matched to self-contained memories.

To score an experiment, save model outputs as JSONL rows such as:

```json
{"id":"005a7cc692bb148f425e204b","question_type":"reference_resolution","prediction":"Model Birch","method":"full_history"}
```

Then run `python score_predictions.py predictions.jsonl`. Use IDs from the current prompt file. The scorer rejects duplicate or unknown question keys and reports coverage and invalid outputs for each method. Accuracy uses submitted rows, with invalid outputs counted as incorrect. False agreement uses valid answers to reference-resolution questions whose expected answer is `UNRESOLVED`.

## Quality gates / important limitations

**Automatically checked:** 80 families, four condition-balanced variants per family, four update blocks of six turns, fixed original mapping within family, one-turn contrast among accepted/rejected/unresolved, one-turn late contrast for accepted/revised, and internally consistent template-derived answer keys.

**Not yet checked:** independent human annotation, whether dialogues sound natural, whether a model interprets every response as intended, whether any domain-level or lexical artifacts enable shortcut learning, actual Qwen token counts, and whether Qwen has headroom below ceiling. These are **synthetic pilot data**, not a final publishable benchmark.

Each dialogue contains 24 turns. Test exact token lengths with the target model tokenizer; the 512-token budget may not impose meaningful compression for every example. The next dataset version should add more varied human-authored paraphrases and harder implicit/partial acknowledgments only after the explicit-evidence baseline has been validated.

**Design caveat:** We deliberately make agreement/rejection explicit in v0.1. A model may find these cases easy with full history; the intended challenge is retaining the agreement across updates and a restricted memory. This does **not** by itself measure general human common ground.
