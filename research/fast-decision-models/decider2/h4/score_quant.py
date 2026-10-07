"""H4: low-bit sensitivity, each model against its own bf16 (laptop, pure python)."""
import sys, json, os
sys.path.insert(0, '~/decider2/evalkit')
import numpy as np
import evalkit as EK
from score_h4 import load, arg

D = '~/decider2/h4/preds/'


def flips(Q, R, suite):
    n = f = 0
    for it in EK.load_suite(suite):
        for q in it['questions']:
            if it['id'] in Q and q in Q[it['id']] and it['id'] in R and q in R[it['id']]:
                n += 1; f += arg(EK._norm(Q[it['id']][q])) != arg(EK._norm(R[it['id']][q]))
    return f / max(n, 1), n


def fg(Q, R, suite):
    """of the pairs the reference tracks, the fraction Q also tracks (the 'fgh' of the brief, against the model's own bf16)"""
    tr = []
    for pr in EK._pairs(suite):
        q = pr['q']
        try:
            r = arg(EK._norm(R[pr['a']][q])) == pr['ea'] and arg(EK._norm(R[pr['b']][q])) == pr['eb']
            p = arg(EK._norm(Q[pr['a']][q])) == pr['ea'] and arg(EK._norm(Q[pr['b']][q])) == pr['eb']
        except KeyError:
            continue
        if r: tr.append(p)
    return float(np.mean(tr)) if tr else float('nan'), len(tr)


refs = {'tt': load(D + 'this-that-model-1.0_sf.jsonl'),
        'hob': {**EK._as_preds('REAL-agree', 'hobson'), **EK._as_preds('CF', 'hobson'), **EK._as_preds('CF-probe', 'hobson')},
        'hob_merged': {**EK._as_preds('REAL-agree', 'merged_full'), **EK._as_preds('CF', 'merged_full'), **EK._as_preds('CF-probe', 'merged_full')}}
out = {}
for m in ('hob', 'tt'):
    for cfg in ('w8a8', 'w4g64'):
        p = D + f'q_{m}_{cfg}.jsonl'
        if not os.path.exists(p): continue
        Q = load(p)
        R = refs[m]
        row = {}
        for s in ('REAL-agree', 'CF', 'CF-probe'):
            row[f'{s}_flips'], row[f'{s}_n'] = flips(Q, R, s)
        for s in ('CF', 'CF-probe'):
            row[f'{s}_fg_own'], _ = fg(Q, R, s)
            sc = EK.score(s, Q, baselines=False)['model']; row[f'{s}_flip'] = sc.get('flip'); row[f'{s}_acc'] = sc.get('acc')
        if m == 'hob':
            row['REAL-agree_flips_vs_merged'], _ = flips(Q, refs['hob_merged'], 'REAL-agree')
            row['CF_fg_vs_merged'], _ = fg(Q, refs['hob_merged'], 'CF')
        ra = EK.score('REAL-agree', Q, baselines=False)['model']; row['REAL_agree_sd_vs_hobson'] = ra.get('agree_sd')
        out[f'{m}_{cfg}'] = row
        print(f'{m}_{cfg}', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
json.dump(out, open('~/decider2/h4/quant_scores.json', 'w'), indent=1)
