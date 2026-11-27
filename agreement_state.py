import json
import re

# Rows are label, candidate, proposer, responder, status, proposal turn, reply turn, revision link.
def apply_events(state, events):
    by_key = {(r[0], r[1]): r.copy() for r in state}
    for label, entity, proposer, responder, status, proposal, reply in events:
        prior = by_key.get((label, entity))
        if status == "P" and prior and prior[4] == "C":
            continue
        revision = "-"
        if status == "C":
            for key, record in by_key.items():
                if key[0] == label and key[1] != entity and record[4] == "C":
                    record[4] = "S"
                    record[7] = str(reply)
                    revision = str(record[5])
        by_key[label, entity] = [label, entity, proposer, responder, status, proposal, reply, revision]
    return list(by_key.values())


def render_state(state):
    if not state:
        return ""
    header = "label|referent|proposer|responder|status|proposal turn|reply turn|revision link\nC confirmed; P pending; R rejected; S superseded.\n"
    return header + "\n".join("|".join(map(str,r)) for r in sorted(state,key=lambda x:(x[5],x[6])))


def fit_state(state, count):
    state = state.copy()
    while count(render_state(state)) > 256:
        drop = min(range(len(state)),key=lambda i: ({"S":0,"R":0,"P":1,"C":2}[state[i][4]],state[i][6]))
        state.pop(drop)
    return state


def prompt_events(state, block):
    return (
        "Extract naming events from the new dialogue block. Ignore logistical talk.\n"
        "For each concrete naming proposal, emit one line: label|exact referent|proposer A or B|"
        "responder A or B|status C/R/P|proposal turn number|reply turn number. "
        "C = the other speaker adopts it, R = rejects it, P = no final decision, including an unaccepted replacement. "
        "Emit a replacement as its own event. Do not repeat events from earlier turns. "
        "Use NONE if no naming event. Return only lines in this format.\n"
        f"Previous agreement record:\n{render_state(state) or 'empty'}\nNew dialogue:\n{block}"
    )


def parse_events(text, block):
    events, invalid = [], 0
    text_by_turn = {int(line.split('.',1)[0]):line.split('. ',1)[1] for line in block.splitlines()}
    for line in text.splitlines():
        line = line.strip().strip('`')
        if not line or line.upper() == 'NONE':
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) != 7:
            invalid += 1
            continue
        label, referent, proposer, responder, status, proposal, reply = parts
        if not (proposal.isdigit() and reply.isdigit()):
            invalid += 1
            continue
        proposal, reply = int(proposal), int(reply)
        ptext, rtext = text_by_turn.get(proposal,''), text_by_turn.get(reply,'')
        if not (label and referent and proposer in ('A','B') and responder in ('A','B')
                and proposer != responder and status in ('C','R','P')
                and ptext.startswith(proposer + ': ') and rtext.startswith(responder + ': ')
                and proposal < reply and label in ptext and referent in ptext):
            invalid += 1
            continue
        events.append((label,referent,proposer,responder,status,proposal,reply))
    return sorted(set(events),key=lambda x:(x[5],x[6],x[0],x[1],x[4])), invalid
