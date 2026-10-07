"""compact results table for the report from preds/*.json (dev rows)."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h1score_copy as S
D = '~/decider2/h3/preds/'
ORDER = [('hobson', 'bf16'), ('hobson', 'w4a4'), ('hobson', 'w4a4r'), ('hobson', 'nvfp4'), ('hobson', 'nvfp4r'), ('hobson', 'g128r'),
         ('v1', 'bf16'), ('v1', 'w4a4'), ('v1', 'w4a4r'), ('v1', 'nvfp4'), ('v2', 'bf16'), ('v2', 'w4a4'), ('v2', 'w4a4r'), ('v2', 'nvfp4'),
         ('qat', 'bf16'), ('qat', 'w4a4'), ('qatnv', 'bf16'), ('qatnv', 'nvfp4'), ('v1x2', 'bf16'), ('v1x2', 'nvfp4'),
         ('hobson', 'w4a4r-qb16'), ('hobson', 'nvfp4-qb16'), ('hobson', 'nvfp4r-qb16'), ('v1', 'nvfp4-qb16'), ('v2', 'w4a4r-qb16'), ('qatnv', 'nvfp4-qb16')]
sub = sys.argv[1] if len(sys.argv) > 1 else 'dev'
print('| arm | format | REAL flips vs hobson | vs own bf16 | REAL agree_sd | CF fgh | CF-probe fgh | JB-hard (McNemar p) | low-bit pass |')
print('|---|---|---|---|---|---|---|---|---|')
out = {}
for tag, cfg in ORDER:
    f = D + f'{tag}__{cfg}__{sub}.json'
    if not os.path.exists(f): continue
    pr = json.load(open(f))
    o = S.row(pr)
    own = D + f'{tag}__bf16__{sub}.json'
    ow = json.load(open(own)) if os.path.exists(own) and cfg != 'bf16' else None
    r = o['REAL-agree']; vo = S.flips(pr, 'REAL-agree', ow)['rate'] if ow else float('nan')
    cf = o.get('CF', {}).get('fgh', float('nan')); cp = o.get('CF-probe', {}).get('fgh', float('nan'))
    jb = o.get('JB-hard', {}); mc = jb.get('mcnemar', {})
    ok = r['flip_rate'] < 0.005 and cf >= 0.97 and cp >= 0.97 and not (mc.get('loss', 0) > mc.get('gain', 0) and mc.get('p', 1) < 0.05)
    vos = f'{100*vo:.1f}%' if vo == vo else '-'
    print(f"| {tag} | {cfg} | {100*r['flip_rate']:.2f}% ({r['flips']}/{r['n']}) | {vos} | {r['agree_sd']:.3f} | {cf:.3f} | {cp:.3f} | {jb.get('acc', float('nan')):.3f} (p {mc.get('p', float('nan'))}) | {'PASS' if ok else 'fail'} |")
    out[f'{tag}::{cfg}'] = dict(real_flips=r['flip_rate'], vs_own=vo, agree_sd=r['agree_sd'], cf_fgh=cf, cfp_fgh=cp, jb=jb.get('acc'), mcnemar=mc)
json.dump(out, open(f'~/decider2/h3/table_{sub}.json', 'w'), indent=1)
