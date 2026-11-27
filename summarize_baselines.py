import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

pilot = Path(sys.argv[1])

def load(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def norm(value):
    return value.strip().strip('`"\' ').rstrip('.').casefold()

key = {(row["id"], row["question_type"]): row for row in load(pilot / "answer_key.jsonl")}
method_data = defaultdict(dict)
for prediction_file in map(Path, sys.argv[2:]):
    for row in load(prediction_file):
        k = row["id"], row["question_type"]
        predictions = method_data[row["method"]]
        assert k in key and k not in predictions
        predictions[k] = row

questions = ("factual_recall", "reference_resolution", "grounding_status")
for method, predictions in method_data.items():
    print(method, len(predictions))
    for question in questions:
        families = defaultdict(list)
        for k, row in predictions.items():
            gold = key[k]
            if k[1] == question:
                families[gold["family_id"]].append(norm(row["prediction"]) == norm(gold["answer"]))
        counts = [(sum(values), len(values)) for values in families.values()]
        right = sum(r for r, n in counts)
        total = sum(n for r, n in counts)
        if total:
            rng = random.Random(20261119)
            samples = []
            for _ in range(10000):
                drawn = rng.choices(counts, k=len(counts))
                samples.append(sum(r for r, n in drawn) / sum(n for r, n in drawn))
            samples.sort()
            print(question, f"{right}/{total}", f"CI [{samples[249]:.4f}, {samples[9749]:.4f}]")

for method, predictions in method_data.items():
    if not method.startswith("qwen"):
        continue
    tokens = sorted(row["prompt_tokens"] for row in predictions.values())
    print("Prompt tokens min, median, max, total", tokens[0],
          (tokens[(len(tokens) - 1) // 2] + tokens[len(tokens) // 2]) / 2,
          tokens[-1], sum(tokens))
    wrong = Counter()
    for k, response in predictions.items():
        gold = key[k]
        answer = norm(response["prediction"])
        if answer != norm(gold["answer"]):
            wrong[k[1], gold["condition"], answer] += 1
    for record, count in wrong.most_common(30):
        print(count, *record)

for filename in ("full_history_prompts.jsonl", "answer_key.jsonl"):
    print(filename, hashlib.sha256((pilot / filename).read_bytes()).hexdigest())
