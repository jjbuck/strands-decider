"""compression when merges are restricted to non-conversation text (KB documents, tool lists, headers); conversation sections stay Qwen."""
import os, sys, json, re, numpy as np
sys.path[:0] = [os.path.expanduser('~/work/j7'), os.path.expanduser('~/work/evalkit')]
from superbpe import Super, qwen_tok
import evalkit as EK
from strands_decider.prompting import render_state
tok = qwen_tok()
S = {lv: Super(os.path.expanduser(f'~/work/j7/tok/sb{lv}.json'), tok=tok) for lv in ('16k', '64k')}
CONV = re.compile(r'CONVERSATION|NEWEST MESSAGE|PROPOSED MESSAGE|CUSTOMER', re.I)


import chan
def protect_msgs(text, offs):
    """True for tokens in user / assistant natural-language lines (chan line roles 1, 2), incl. continuation lines"""
    ann = chan.annotate(text); r = ann['role'] % 6
    return [bool(r[min(max(b - 1, 0), len(text) - 1)] in (1, 2)) if b > a else False for a, b in offs]


def protect_mask(text, offs):
    """True for tokens inside a conversation-like section"""
    marks = np.zeros(len(text) + 1, dtype=bool)
    hdr = list(re.finditer(r'^--- (.+?) ---\s*$', text, re.M))
    for i, m in enumerate(hdr):
        end = hdr[i + 1].start() if i + 1 < len(hdr) else len(text)
        if CONV.search(m.group(1)): marks[m.end():end] = True
    return [bool(marks[min(a, len(text))]) if b > a else False for a, b in offs]


res = {}
for suite in ('REAL-agree', 'LONG', 'CF', 'CF-probe'):
    its = EK.load_suite(suite)
    tot = dict(q=0); fr = []
    for lv in S: tot[lv] = 0; tot[lv + 'p'] = 0; tot[lv + 'm'] = 0
    fm = []
    for it in its:
        t = render_state(it['state']); enc = tok(t, add_special_tokens=True, return_offsets_mapping=True)
        ids = enc['input_ids']; pm = protect_mask(t, enc['offset_mapping']); pu = protect_msgs(t, enc['offset_mapping'])
        tot['q'] += len(ids); fr.append(np.mean(pm)); fm.append(np.mean(pu))
        for lv, s_ in S.items():
            tot[lv] += len(s_.merge_ids(ids)[0]); tot[lv + 'p'] += len(s_.merge_ids(ids, pm)[0]); tot[lv + 'm'] += len(s_.merge_ids(ids, pu)[0])
    res[suite] = {k: round(tot['q'] / v, 3) for k, v in tot.items() if k != 'q'}
    res[suite]['conv_token_frac'] = round(float(np.mean(fr)), 3); res[suite]['msg_token_frac'] = round(float(np.mean(fm)), 3)
    print(suite, res[suite], flush=True)
json.dump(res, open(os.path.expanduser('~/work/j7/res/protstat.json'), 'w'), indent=1)
