#!/usr/bin/env python3
"""Generate a reproducible development-only common-ground compression pilot.

No model calls are made here. Scenario metadata and answers MUST NOT be shown
in the evaluation prompt. The split is always `development` by default.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

DOMAIN_CONFIGS = [
    ("model comparison", "Model", ["Birch", "Cedar", "Pine", "Maple", "Elm", "Spruce", "Willow", "Aspen"], "benchmark runs"),
    ("shared drive cleanup", "Folder", ["North", "South", "East", "West", "Archive", "Current", "Review", "Draft"], "file organization"),
    ("experiment planning", "Experiment", ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"], "experiment tracking"),
    ("dashboard review", "Chart", ["Revenue", "Latency", "Traffic", "Errors", "Retention", "Usage", "Coverage", "Volume"], "dashboard notes"),
    ("campus event logistics", "Room", ["Oak", "Cypress", "Laurel", "Walnut", "Juniper", "Magnolia", "Hickory", "Poplar"], "event logistics"),
    ("codebase refactor", "Branch", ["Copper", "Silver", "Indigo", "Amber", "Teal", "Olive", "Coral", "Violet"], "development work"),
    ("prototype comparison", "Prototype", ["Small", "Medium", "Large", "Compact", "Wide", "Tall", "Light", "Dense"], "design review"),
    ("data audit", "Dataset", ["Training", "Validation", "Control", "Pilot", "Audit", "Holdout", "Review", "Backup"], "data auditing"),
    ("technical report", "Document", ["Outline", "Methods", "Results", "Appendix", "Summary", "Draft", "Notes", "Review"], "report preparation"),
    ("workflow planning", "Workflow", ["Intake", "Verify", "Review", "Approve", "Archive", "Notify", "Route", "Deploy"], "workflow coordination"),
    ("software testing", "Test", ["Smoke", "Load", "Unit", "API", "Client", "Server", "Integration", "Regression"], "test planning"),
    ("research meeting", "Slide", ["Intro", "Methods", "Goals", "Risks", "Timeline", "Results", "Questions", "Summary"], "meeting preparation"),
]
LABELS = ["Falcon", "Atlas", "Nimbus", "Saffron", "Orion", "Harbor", "Beacon", "Kestrel", "Echo", "Cobalt", "Juniper", "Meridian", "Quill", "Summit", "Lumen", "Cascade", "Ember", "Nova", "Clover", "Slate", "Prism", "Citrine", "Drift", "Aurora", "Pioneer", "Ridge", "Willow", "Canyon", "Tangent", "Anchor", "Lyra", "Crescent"]
VARIANTS = ("accepted", "rejected", "unresolved", "revised")
TARGET_SLOTS = (4, 10, 14)

OPENINGS = [
    "We need a consistent set of short names before finishing today's {task}.",
    "Let's settle some internal labels while we work through the {task}.",
    "I want our notes for the {task} to use predictable names.",
    "We should agree on the terms we'll use in the {task} discussion.",
]
OPENING_REPLIES = [
    "That makes sense. There are several similar items on the list, so labels might help.",
    "Agreed on the goal. We should still be careful about what each label names.",
    "I'm following. I'll keep a running list as we make decisions.",
    "Okay. We can resolve each name as we reach the corresponding item.",
]
ACCEPT_TEMPLATES = [
    "I agree. For this discussion, {label} will mean {old}.",
    "Yes, let's adopt {label} as our shared name for {old}.",
    "That works for me. We will both use {label} for {old}.",
    "Confirmed. When either of us says {label}, we mean {old}.",
    "That's the name I'll put in our notes: {label} for {old}.",
    "Let's go with {label} for {old}; I'll use it too.",
    "We're settled on {label} for {old}.",
    "I'll use {label} for {old} as well.",
]
REJECT_TEMPLATES = [
    "No, I don't agree to using {label} for {old}. Let's leave that name unused.",
    "I reject that proposal. We should not adopt {label} for {old}.",
    "Let's not make that agreement; I don't want {label} to mean {old}.",
    "No. I am explicitly declining the proposal to call {old} {label}.",
    "Please drop that suggestion. I don't want us to use {label} for {old}.",
    "Let's rule out that name for {old}.",
    "I won't use {label} for {old}.",
    "Take that proposal off the list. I don't want {label} for {old}.",
]
UNRESOLVED_TEMPLATES = [
    "I understand your suggestion, but I haven't agreed. Let's decide on {label} later.",
    "I heard the proposed name. I'd prefer to leave {label} undecided for now.",
    "Maybe, but please don't treat {label} as agreed yet. We'll revisit it.",
    "I can consider it, but I am not confirming that name at this point.",
    'Can we come back to the name after we check the notes?',
    'Keep {label} on the list of options for now.',
    'I need time to think about that name before we decide.',
    'Leave the name open until our next pass.',
]
NEW_PROPOSALS = [
    "One alternative proposal: could {label} refer to {new}? I am not deciding it alone.",
    "Regardless of earlier suggestions, could we now agree to use {label} for {new}?",
    "I want to propose a different mapping: {label} for {new}. Do you accept?",
]
PENDING_REPLIES = [
    "Let's hold off. I'm not agreeing to use {label} for {new} yet.",
    "I heard the alternative, but I am not accepting {label} for {new} at this point.",
    "I'd rather leave the proposal about {label} pending until we discuss it further.",
    "Let's leave that alternative open until our next discussion.",
    'Keep that as an option for now.',
    'Can we decide on the alternative after we review the notes?',
]
REVISION_REPLIES = [
    "Yes, I agree to replace the earlier mapping. From now on, {label} means {new}.",
    "Confirmed. We both agree that {label} now refers to {new}, replacing the earlier mapping.",
    "Let's make that revision official: {label} is now our shared name for {new}.",
    "I'll switch to {label} for {new} too.",
    "Update our notes. We'll both use {label} for {new} from here on.",
    "Let's switch. Use {label} for {new} in the shared notes.",
]


def make_turns(rng, cfg, labels, entities, old, new, target_slot, variant, proposer):
    domain, kind, _, task = cfg
    other = "B" if proposer == "A" else "A"
    speakers = [proposer if i % 2 == 0 else other for i in range(24)]
    side1, side2, side3 = labels[1:]
    side1_entity, side2_entity, side3_entity = entities[3], entities[4], entities[0]
    kind_plural = kind.lower() + "s"

    # Each slot describes a plausible utterance. The target proposal/reply replace
    # one neutral exchange. All three non-revision variants differ in ONE turn only.
    turn_text = [
        rng.choice(OPENINGS).format(task=task),
        rng.choice(OPENING_REPLIES),
        f"I've put the candidate {kind_plural} into the shared planning sheet so we can compare them side by side.",
        f"I see the entries. It will help if we distinguish decisions from suggestions while we work.",
        f"For now, let's compare the {kind.lower()} entries by their current descriptions, not by any nickname.",
        f"That seems sensible. We should only use a nickname after we both settle on it.",
        f"I checked the project notes from yesterday and marked which items still need another pass.",
        f"Thanks. Please make sure the notes don't silently turn tentative labels into final decisions.",
        f"Can we call {side1} the {side1.lower()} shorthand for {old.replace(kind+' ', kind+' ')}? Actually, let's use {side1} for the other entry named later.",
        f"I follow the distinction. We can settle the first secondary label in the next pass.",
        f"There's also a separate issue about the {task}; I'll flag it without changing any existing names.",
        f"Good idea. Let's keep that issue on the list until we check the remaining details.",
        f"Let's adopt {side2} as our name for the second supporting item; it won't change the earlier labels.",
        f"Agreed. {side2} is established only for that supporting item.",
        f"I added a note comparing the alternatives, including the relevant constraints from the project brief.",
        f"I read it. We can keep the comparison separate from our naming decisions.",
        f"We still need to check consistency across the project notes and the discussion transcript.",
        f"I'll check both sources and make sure unresolved naming questions remain marked as open.",
        f"For another agreed shorthand, let's use {side3} for the item we'll track in the follow-up notes.",
        f"Yes, let's use {side3} for that separate tracking item.",
        "We should revisit one of the labels only if we explicitly discuss a replacement.",
        "Agreed. A suggested alternative should not automatically override a current decision.",
        f"Let's finish this pass by checking that every entry has a clear explanation in the shared notes.",
        f"I can do that. The remaining tasks do not require changing the naming conventions again.",
    ]
    # Make the secondary conventions real and checkable (no conflicting semantics).
    turn_text[8] = f"For a separate entry, could we use {side1} as the short name for {side1_entity}?"
    turn_text[9] = f"Yes. I agree that {side1} will refer to {side1_entity}."
    turn_text[12] = f"I propose we call {side2_entity} {side2}; that is a separate naming decision."
    turn_text[13] = f"Agreed. {side2} refers to {side2_entity} for both of us."
    turn_text[18] = f"For our final secondary label, could {side3} refer to {side3_entity}?"
    turn_text[19] = f"Yes. We both agree that {side3} names {side3_entity}."

    label = labels[0]
    turn_text[target_slot] = rng.choice([
        f"Could we use {label} as the shared name for {old} during this {task}?",
        f"I propose we call {old} {label} in this conversation. Do you agree?",
        f"For the sake of our joint notes, my suggestion is {label} for {old}. Is that okay?",
        f"I'd like us to use {label} to refer to {old}. Are you willing to adopt that?",
    ])
    if variant in ("accepted", "revised"):
        turn_text[target_slot + 1] = rng.choice(ACCEPT_TEMPLATES).format(label=label,old=old)
    elif variant == "rejected":
        turn_text[target_slot + 1] = rng.choice(REJECT_TEMPLATES).format(label=label,old=old)
    elif variant == "unresolved":
        turn_text[target_slot + 1] = rng.choice(UNRESOLVED_TEMPLATES).format(label=label,old=old)
    # All variants contain a later proposal, accepted only in the revised variant.
    # This explicitly tests that proposing an alternative does NOT overwrite shared state.
    proposal_txt = rng.choice(NEW_PROPOSALS).format(label=label,new=new,old=old)
    turn_text[20] = proposal_txt
    if variant == "revised":
        turn_text[21] = rng.choice(REVISION_REPLIES).format(label=label,new=new,old=old)
    else:
        turn_text[21] = rng.choice(PENDING_REPLIES).format(label=label,new=new)
    turns = [{"turn":i+1,"speaker":speakers[i],"text":turn_text[i]} for i in range(24)]
    assert all(t['text'] for t in turns)
    return turns


def make_family(rng, i):
    cfg = DOMAIN_CONFIGS[i % len(DOMAIN_CONFIGS)]
    _, kind, names, _ = cfg
    sampled_names = rng.sample(names, 5)
    entities = [f"{kind} {name}" for name in sampled_names]
    labels = rng.sample(LABELS,4)
    family_id = f"dev_family_{i+1:03d}"
    old, new = entities[1], entities[2]
    # Fix all phrasing decisions across variants except the target response
    # and late response: draw once from family RNG; we enforce common turns.
    family_seed = rng.randrange(1, 10**9)
    design_rng = random.Random(f"design:{family_seed}")
    proposer = design_rng.choice(("A", "B"))
    target_slot = design_rng.choice(TARGET_SLOTS)
    candidates = entities.copy()
    design_rng.shuffle(candidates)
    dialogues = []
    for v in VARIANTS:
        vrng = random.Random(family_seed)
        turns = make_turns(vrng, cfg, labels, entities, old, new, target_slot, v, proposer)
        target_label = labels[0]
        active = old if v == 'accepted' else new if v == 'revised' else 'UNRESOLVED'
        original_status = {"accepted":"ACCEPTED", "rejected":"REJECTED", "unresolved":"PENDING", "revised":"SUPERSEDED"}[v]
        blocks = [turns[j:j+6] for j in range(0,24,6)]
        record_id = hashlib.sha256(
            f"{family_id}:{family_seed}:{v}".encode()
        ).hexdigest()[:24]
        item = {
            "id":record_id,
            "family_id":family_id,
            "split":"development",
            "condition":v,
            "domain":cfg[0],
            "participants":["A","B"],
            "target_label":target_label,
            "entities":candidates,
            "dialogue":turns,
            "update_blocks":blocks,
            "gold":{
                "original_proposed_referent":old,
                "current_shared_referent":active,
                "original_proposal_final_status":original_status,
                "proposed_replacement":new,
                "replacement_accepted":v == 'revised',
            },
            "questions":[
                {"type":"factual_recall", "text":f"Which {kind.lower()} did Speaker {proposer} first suggest using the label '{target_label}' for? Answer with one candidate entity exactly.","answer":old},
                {"type":"reference_resolution", "text":f"At the end of this conversation, which {kind.lower()} do Speakers A and B mutually use '{target_label}' to refer to? Answer with one candidate entity exactly, or UNRESOLVED if no shared convention has been established.","answer":active},
                {"type":"grounding_status", "text":f"What is the final status of the first proposal for the label '{target_label}'? Choose one: ACCEPTED, REJECTED, PENDING, SUPERSEDED.","answer":original_status},
            ],
            "evidence_turns":{
                "initial_proposal":target_slot+1,
                "initial_response":target_slot+2,
                "alternative_proposal":21,
                "alternative_response":22,
            },
            "metadata_only":"Do not include condition, gold, questions[].answer, or evidence_turns in model input.",
        }
        dialogues.append(item)
    return dialogues


def verify_family(items):
    assert len(items)==4
    by = {x['condition']:x for x in items}
    assert set(by)==set(VARIANTS)
    assert len({x['target_label'] for x in items})==1
    assert len({tuple(x['entities']) for x in items})==1
    assert all(len(x['dialogue'])==24 and len(x['update_blocks'])==4 for x in items)
    # In the first three variants, exactly one utterance differs, namely the response.
    a,r,u = (by[k] for k in ('accepted','rejected','unresolved'))
    target_idx=a['evidence_turns']['initial_response']-1
    for x in (r,u):
        diff = [i for i in range(24) if a['dialogue'][i]!=x['dialogue'][i]]
        assert diff==[target_idx],(a['family_id'],diff,target_idx)
    # Revised differs from accepted only in the later acceptance response.
    revised=by['revised']
    diff = [i for i in range(24) if a['dialogue'][i]!=revised['dialogue'][i]]
    assert diff==[21],(a['family_id'],diff)
    assert all(x['questions'][0]['answer']==a['questions'][0]['answer'] for x in items)
    assert [by[k]['gold']['current_shared_referent'] for k in VARIANTS] == [a['gold']['original_proposed_referent'],'UNRESOLVED','UNRESOLVED',a['gold']['proposed_replacement']]
    for item in items:
        assert [len(b) for b in item['update_blocks']] == [6,6,6,6]
        assert [t['turn'] for t in item['dialogue']] == list(range(1,25))
    return True


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--families',type=int,default=80)
    p.add_argument('--seed',type=int,default=20261119)
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
    args=p.parse_args()
    if args.families < 1:
        p.error('--families must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    rng=random.Random(args.seed)
    dialogues=[]
    for i in range(args.families):
        f=make_family(rng,i)
        verify_family(f)
        dialogues.extend(f)
    # Dataset JSONL, machine-readable and complete.
    with (args.out/'pilot_development.jsonl').open('w',encoding='utf-8') as f:
        for item in dialogues:
            f.write(json.dumps(item,ensure_ascii=False)+'\n')
    # Compact review CSV, model input NOT to be constructed from these gold columns.
    fields=['id','family_id','condition','domain','target_label','original_proposed_referent','current_shared_referent','original_proposal_final_status','proposed_replacement','initial_proposal_turn','initial_response_turn','replacement_proposal_turn','replacement_response_turn','dialogue_text']
    with (args.out/'pilot_review.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for d in dialogues:
            writer.writerow({
                'id':d['id'],'family_id':d['family_id'],'condition':d['condition'],'domain':d['domain'],'target_label':d['target_label'],
                'original_proposed_referent':d['gold']['original_proposed_referent'],
                'current_shared_referent':d['gold']['current_shared_referent'],
                'original_proposal_final_status':d['gold']['original_proposal_final_status'],
                'proposed_replacement':d['gold']['proposed_replacement'],
                'initial_proposal_turn':d['evidence_turns']['initial_proposal'],
                'initial_response_turn':d['evidence_turns']['initial_response'],
                'replacement_proposal_turn':d['evidence_turns']['alternative_proposal'],
                'replacement_response_turn':d['evidence_turns']['alternative_response'],
                'dialogue_text':'\n'.join(f"{t['speaker']}: {t['text']}" for t in d['dialogue'])
            })
    # Annotation sheet: no printed gold answers for independent review.
    with (args.out/'human_annotation.csv').open('w',encoding='utf-8',newline='') as f:
        fieldnames=['id','transcript','annotator','original_proposed_referent','current_shared_referent','original_proposal_status','evidence_turns','ambiguous','notes']
        writer=csv.DictWriter(f,fieldnames=fieldnames)
        writer.writeheader()
        annotation_order = sorted(dialogues, key=lambda d: d['id'])
        for d in annotation_order:
            writer.writerow({'id':d['id'],
              'transcript':'\n'.join(f"{t['turn']}. {t['speaker']}: {t['text']}" for t in d['dialogue']),
              'annotator':'','original_proposed_referent':'','current_shared_referent':'',
              'original_proposal_status':'','evidence_turns':'','ambiguous':'','notes':''})
    with (args.out/'pilot_first_family.jsonl').open('w',encoding='utf-8') as f:
        for d in dialogues[:4]:
            f.write(json.dumps(d, ensure_ascii=False)+'\n')
    # Tiny preview that's easy to paste or inspect without the JSONL.
    with (args.out/'preview_examples.txt').open('w',encoding='utf-8') as f:
        for d in dialogues[:4]:
            f.write(f"===== {d['family_id']} | {d['condition']} | {d['id']} =====\n")
            for turn in d['dialogue']:
                f.write(f"{turn['turn']:02d}. {turn['speaker']}: {turn['text']}\n")
            for q in d['questions']:
                f.write(f"{q['type']}: {q['text']}\n  GOLD: {q['answer']}\n")
            f.write('\n')
    stats={
        'seed':args.seed, 'families':args.families,'dialogues':len(dialogues),'questions':sum(len(d['questions']) for d in dialogues),
        'condition_counts':dict(Counter(d['condition'] for d in dialogues)),
        'dialogue_turns':24,'update_blocks':4,'turns_per_block':6,
        'split':'development',
        'pair_validation':'passed: accepted/rejected/unresolved differ in only the initial response; accepted/revised differ only in replacement response',
        'caution':'Gold states are template-derived; manually review before using as final empirical labels.',
    }
    (args.out/'manifest.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
