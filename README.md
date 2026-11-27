# Preserving Conversational Agreements under Memory Compression

Research proposal, synthetic development pilot, Colab notebooks, and experiment results.

- [Research proposal](Preserving_Conversational_Agreements_under_Memory_Compression.pdf)
- [Pilot dataset and scoring instructions](PILOT.md)
- [Full-history notebook](qwen_colab_baseline.ipynb) and [scores](qwen_colab_scores.txt)
- [Extractive memory notebook](qwen_compressed_memory.ipynb) and [paired comparison](qwen_compressed_comparison.txt)
- [Agreement memory notebook](agreement_memory_tests.ipynb), [scores](agreement_tests_scores.txt), and [paired comparison](agreement_tests_comparison.txt)

The pilot contains 320 template-generated dialogues from 80 development families. The three notebook experiments use the same frozen Qwen2.5-7B-Instruct checkpoint. The answer key and oracle states support offline scoring and diagnosis.

Score predictions with the answer key in this directory:

```sh
python3 score_predictions.py agreement_tests_predictions.jsonl
```
