"""Run two deterministic controls using only model-facing inputs."""
import json
import re
import sys
from pathlib import Path

prompts = Path(sys.argv[1])
output = Path(sys.argv[2])
revision_openers = (
    "Yes, I agree to replace ",
    "Confirmed. We both agree ",
    "Let's make that revision official:",
    "I'll switch to ",
    "Update our notes. We'll both use ",
    "Let's switch.",
)
with prompts.open() as source, output.open("w") as result:
    for line in source:
        row = json.loads(line)
        stem, question = row["prompt"].rsplit("\n\nQUESTION:\n", 1)
        dialogue, choices = stem.rsplit("\n\nCANDIDATE ENTITIES:\n", 1)
        entities = [choice.removeprefix("- ") for choice in choices.splitlines()]
        mode = row["question_type"]
        answer = {
            "factual_recall": entities[0],
            "reference_resolution": "UNRESOLVED",
            "grounding_status": "ACCEPTED",
        }[mode]
        last_block = dialogue.splitlines()[-6:]
        replacement, reply = last_block[2:4]
        revised = reply.split(": ", 1)[1].startswith(revision_openers)
        revision_answer = answer
        if revised and mode == "reference_resolution":
            revision_answer, = [entity for entity in entities if entity in replacement]
        if revised and mode == "grounding_status":
            revision_answer = "SUPERSEDED"
        for method, prediction in (
            ("no_dialogue_rule", answer),
            ("last_block_rule", revision_answer),
        ):
            result.write(json.dumps({
                "id": row["id"],
                "question_type": mode,
                "prediction": prediction,
                "method": method,
            }) + "\n")
