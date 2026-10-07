"""collect per-checkpoint eval metrics (quick subset: REAL-agree-SD, CF-T, CF-probe-T) into res/curve_eval.json"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import h1score as S
R = os.path.expanduser('~/decider2/h1/res/'); G = os.path.expanduser('~/decider2/h1/g2res/')
def pt(P, step):
    o = S.row(P); r = o.get('REAL-agree', {})
    return dict(step=step, real_flips=r.get('flip_rate'), tv=r.get('tv'), agree_sd=r.get('agree_sd'), cf_fgh=o.get('CF', {}).get('fgh'), cfp_fgh=o.get('CF-probe', {}).get('fgh'))
out = {}
if os.path.exists(R + 'e4_w4a8curve.json'):
    P = json.load(open(R + 'e4_w4a8curve.json'))['preds']; pts = []
    for c, st in (('w4a8,gptq', 0), ('w4a8,gptq,lrot=lora_qat_w4a8_B_s300.pt', 300), ('w4a8,gptq,lrot=lora_qat_w4a8_B_s600.pt', 600), ('w4a8,gptq,lrot=lora_qat_w4a8_B_s900.pt', 900)):
        if P.get(c) and sum(len(v) for v in P[c].values()) >= 770: pts.append(pt(P[c], st))
    out['H1 QAT-B W4A8 (GPTQ init)'] = pts
pts = []
for f, st in (('qe0.json', 0), ('qe500.json', 500), ('qe1000.json', 1000), ('qe1500.json', 1500)):
    P = json.load(open(G + f))['preds']['Q:r4c']
    sd = json.load(open(os.path.expanduser('~/decider2/recovered/g2/g2/subsets.json')))['REAL-agree-SD']; keep = {(a[1], a[2]) for a in sd}
    real = S.ids_of('REAL-agree')
    P = {i: {q: v for q, v in qs.items() if (i, q) in keep or i not in real} for i, qs in P.items()}
    pts.append(pt(P, st))
out['G2 QAT W4A4 r4c (RTN init)'] = pts
json.dump(out, open(R + 'curve_eval.json', 'w'), indent=1)
for k, v in out.items():
    print(k)
    for p in v: print('  ', {a: (round(b, 4) if isinstance(b, float) else b) for a, b in p.items()})
