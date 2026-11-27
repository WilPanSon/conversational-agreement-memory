import statistics
from collections import Counter

oracle = json.loads(gzip.decompress(base64.b64decode(GOLD_DATA)))
budget = 256
batch_size = 8
started = time.monotonic()
tokenized = {}
generation_counts = {}


def count(text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def generate(prompts, limit, phase):
    unique = sorted(set(prompts), key=lambda p: (len(p), p))
    generation_counts[phase] = len(unique)
    outputs = {}
    for i in range(0,len(unique),batch_size):
        batch=unique[i:i+batch_size]
        texts=[tokenizer.apply_chat_template([{"role":"user","content":p}],
                 tokenize=False,add_generation_prompt=True) for p in batch]
        inputs=tokenizer(texts,padding=True,add_special_tokens=False,return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated=model.generate(**inputs, do_sample=False,max_new_tokens=limit,
                       temperature=None,top_p=None,top_k=None,pad_token_id=tokenizer.pad_token_id)
        continuation=generated[:,inputs.input_ids.size(1):]
        for prompt, answer, length in zip(batch,tokenizer.batch_decode(continuation,skip_special_tokens=True),
                                          inputs.attention_mask.sum(1).tolist()):
            outputs[prompt]=answer.strip()
            tokenized[prompt]=length
        if (i // batch_size) % 12 == 0 or i+len(batch)==len(unique):
            print(f"{phase}: {i+len(batch)}/{len(unique)} ({time.monotonic()-started:.0f}s)",flush=True)
    return [outputs[p] for p in prompts]


def fit_prose(text):
    ids=tokenizer.encode(text,add_special_tokens=False)
    clipped=len(ids)>budget
    while len(ids)>budget:
        text=tokenizer.decode(ids[:budget],skip_special_tokens=True).strip()
        ids=tokenizer.encode(text,add_special_tokens=False)
    return text,clipped


dialogues={}
for row in rows:
    transcript=row["prompt"].split("\n\nDIALOGUE:\n",1)[1].split("\n\nCANDIDATE ENTITIES:\n",1)[0]
    if row["id"] in dialogues:
        continue
    turns=[f"{i}. {line}" for i,line in enumerate(transcript.splitlines(),1)]
    assert len(turns)==24
    dialogues[row["id"]]=["\n".join(turns[i:i+6]) for i in range(0,24,6)]
assert len(dialogues)==len(oracle)==320

print("GPU:",torch.cuda.get_device_name(),flush=True)
ids=list(dialogues)
prose={i:"" for i in ids}
predicted={i:[] for i in ids}
gold_state={i:[] for i in ids}
diagnostics=Counter()
with Path("agreement_tests_memories.jsonl").open("w", buffering=1) as log:
    for step in range(4):
        blocks=[dialogues[i][step] for i in ids]
        text_prompts=[
            "Update the running dialogue summary using only the previous summary and the new block. "
            "Keep all concrete labels and their proposed referents, speakers, confirmations, rejections, "
            "pending alternatives and revisions. Keep an earlier confirmed meaning active until the other "
            "speaker accepts a replacement. Preserve the original proposal when possible. Omit logistics. "
            "Write concise prose only, under 256 tokens.\n\n"
            f"Previous summary:\n{prose[i] or 'empty'}\n\nNew dialogue:\n{block}"
            for i,block in zip(ids, blocks)]
        for i,text in zip(ids,generate(text_prompts,256,f"prose block {step+1}")):
            prose[i],clipped=fit_prose(text)
            diagnostics["prose_over_budget"]+=clipped
        event_prompts=[prompt_events(predicted[i],block) for i,block in zip(ids,blocks)]
        for i,block,text in zip(ids,blocks,generate(event_prompts,140,f"events block {step+1}")):
            events, invalid=parse_events(text,block)
            expected=[event for event in oracle[i] if step*6 < event[5] <= (step+1)*6]
            # Oracle is accessed only after predicted events have been extracted.
            pred_set={tuple(event) for event in events}
            gold_set={tuple(event) for event in expected}
            diagnostics["event_true_positive"]+=len(pred_set & gold_set)
            diagnostics["event_predicted"]+=len(pred_set)
            diagnostics["event_gold"]+=len(gold_set)
            diagnostics["unparseable_event_lines"]+=invalid
            predicted[i]=fit_state(apply_events(predicted[i],events),count)
            gold_state[i]=fit_state(apply_events(gold_state[i],expected),count)
            log.write(json.dumps({"id":i,"block":step+1,"prose":prose[i],
                       "predicted_state":render_state(predicted[i]),"gold_state":render_state(gold_state[i]),
                       "raw_event_output":text,"parsed_events":events,"gold_events":expected})+"\n")

memory={
    "grounding_aware_prose":prose,
    "predicted_agreement_state":{i:render_state(state) for i,state in predicted.items()},
    "gold_agreement_state":{i:render_state(state) for i,state in gold_state.items()},
}
jobs=[]
for method, memories in memory.items():
    for row in rows:
        text=memories[row["id"]]
        length=count(text)
        assert length <= budget
        before, rest=row["prompt"].split("\n\nDIALOGUE:\n",1)
        _, suffix=rest.split("\n\nCANDIDATE ENTITIES:\n",1)
        before=before.replace("Read the dialogue","Read the dialogue memory")
        prompt=before+"\n\nDIALOGUE MEMORY:\n"+text+"\n\nCANDIDATE ENTITIES:\n"+suffix
        jobs.append((method,row,length,prompt))

prompts=[job[-1] for job in jobs]
with Path("agreement_tests_predictions.jsonl").open("w",buffering=1) as output:
    for (method,row,length,prompt),answer in zip(jobs,generate(prompts,24,"answering")):
        output.write(json.dumps({"id":row["id"],"question_type":row["question_type"],
                     "prediction":answer,"method":method,"memory_tokens":length,
                     "prompt_tokens":tokenized[prompt]})+"\n")
metadata={
    "model":model_id,"revision":revision,"quantization":"bitsandbytes NF4, float16 compute",
    "gpu":torch.cuda.get_device_name(),"torch":torch.__version__,"transformers":transformers.__version__,
    "bitsandbytes":bitsandbytes.__version__,"batch_size":batch_size,"do_sample":False,
    "memory_budget":budget,"update_blocks":4,"answer_max_new_tokens":24,
    "prompt_file_sha256":hashlib.sha256(raw).hexdigest(),"unique_generations":generation_counts,
    "diagnostics":dict(diagnostics),"elapsed_seconds":round(time.monotonic()-started),
    "memory_tokens_mean":{method:round(statistics.mean(map(count,m.values())),2) for method,m in memory.items()}
}
print(json.dumps(metadata,indent=2))
Path("agreement_tests_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
