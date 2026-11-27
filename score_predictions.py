#!/usr/bin/env python3
"""Score JSONL predictions with id, question_type, prediction, and optional method."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
STATUSES = {"ACCEPTED", "REJECTED", "PENDING", "SUPERSEDED"}


def norm(value):
    return str(value).strip().strip('`"\' ').rstrip('.').casefold()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    answers = map(json.loads, (BASE / "answer_key.jsonl").read_text().splitlines())
    keys = {(row["id"], row["question_type"]): row for row in answers}
    dialogues = map(json.loads, (BASE / "pilot_development.jsonl").read_text().splitlines())
    candidates = {row["id"]: {norm(entity) for entity in row["entities"]} for row in dialogues}
    statuses = {norm(status) for status in STATUSES}
    allowed = {
        "factual_recall": lambda record_id: candidates[record_id],
        "reference_resolution": lambda record_id: candidates[record_id] | {"unresolved"},
        "grounding_status": lambda record_id: statuses,
    }
    rows_by_method = defaultdict(dict)
    with args.predictions.open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["id"], row["question_type"]
            method = row.get("method", "default")
            if key not in keys:
                parser.error(f"Unknown prediction {key}")
            if key in rows_by_method[method]:
                parser.error(f"Duplicate prediction for {method}: {key}")
            rows_by_method[method][key] = row.get("prediction")

    if not rows_by_method:
        parser.error("No predictions supplied")

    for method, predictions in sorted(rows_by_method.items()):
        scores = defaultdict(lambda: [0, 0, 0, 0])
        false_agreements = valid_unresolved = invalid_unresolved = 0
        for key, gold in keys.items():
            record_id, question_type = key
            present = key in predictions
            answer = norm(predictions[key]) if present else ""
            valid = present and answer in allowed[question_type](record_id)
            correct = valid and answer == norm(gold["answer"])
            invalid = present and not valid
            for scope in ("all", f"condition:{gold['condition']}"):
                totals = scores[question_type, scope]
                totals[0] += correct
                totals[1] += present
                totals[2] += invalid
                totals[3] += 1

            if question_type == "reference_resolution" and gold["answer"] == "UNRESOLVED":
                valid_unresolved += valid
                false_agreements += valid and answer != "unresolved"
                invalid_unresolved += invalid

        print(f"\nMETHOD: {method}")
        print(f"Coverage: {len(predictions)}/{len(keys)} ({len(keys) - len(predictions)} missing)")
        for (question_type, scope), (correct, present, invalid, expected) in sorted(scores.items()):
            accuracy = f"{correct / present:.3f}" if present else "n/a"
            print(f"{question_type:22s} {scope:20s}: {correct}/{present} = {accuracy}"
                  f" | coverage {present}/{expected} | invalid {invalid}")
        rate = f"{false_agreements / valid_unresolved:.3f}" if valid_unresolved else "n/a"
        print(f"False agreement rate (valid outputs): {false_agreements}/{valid_unresolved} = {rate}"
              f" | invalid {invalid_unresolved}")


if __name__ == "__main__":
    main()
