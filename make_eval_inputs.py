#!/usr/bin/env python3
"""Produce strictly separated, model-facing prompts and private answer keys.

Run: python make_eval_inputs.py
Do not feed answer_key.jsonl or the labeled development file into the model.
"""
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent

def main():
    examples=[json.loads(x) for x in (BASE/'pilot_development.jsonl').read_text().splitlines() if x.strip()]
    prompts=[]
    answers=[]
    for ex in sorted(examples, key=lambda example: example["id"]):
        turns='\n'.join(f"{t['speaker']}: {t['text']}" for t in ex['dialogue'])
        options='\n'.join(f"- {x}" for x in ex['entities'])
        for q in ex['questions']:
            prompt=(
                "Read the dialogue and answer the question using only what the speakers established in their conversation.\n"
                "Do not assume a name was agreed upon just because one speaker proposed it.\n"
                "Provide only the requested answer, with no explanation.\n\n"
                f"DIALOGUE:\n{turns}\n\nCANDIDATE ENTITIES:\n{options}\n\n"
                f"QUESTION:\n{q['text']}\n"
            )
            prompts.append({"id":ex['id'],"question_type":q['type'],"prompt":prompt})
            answers.append({"id":ex['id'],"question_type":q['type'],"answer":q['answer'],"family_id":ex['family_id'],"condition":ex['condition']})
    with (BASE/'full_history_prompts.jsonl').open('w') as f:
        for x in prompts:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    with (BASE/'answer_key.jsonl').open('w') as f:
        for x in answers:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    print('Prompts:',len(prompts),'Answer keys:',len(answers))
if __name__=='__main__':main()
