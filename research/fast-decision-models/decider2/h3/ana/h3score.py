"""H3 laptop scorer over preds/<tag>__<cfg>__<sub>.json (flat {id: {q: dist}}). Uses H1's metric code (copied from h1/code/h1score.py at 22:25 PDT)
so H1 and H3 numbers are computed identically. Flips are vs bf16 hobson refs and vs H3's own in-runtime bf16 (hobson__bf16__<sub>).
python3 h3score.py [glob]"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h1score_copy as S
D = os.path.expanduser('~/decider2/h3/preds/')
pat = sys.argv[1] if len(sys.argv) > 1 else '*'
res = {}
for f in sorted(glob.glob(D + pat + '.json')):
    tag, cfg, sub = os.path.basename(f)[:-5].split('__')
    pr = json.load(open(f))
    dfile = D + f'hobson__bf16__{sub}.json'
    dense = json.load(open(dfile)) if os.path.exists(dfile) and not (tag == 'hobson' and cfg == 'bf16') else None
    o = S.row(pr, dense)
    own = D + f'{tag}__bf16__{sub}.json'
    if tag != 'hobson' and cfg != 'bf16' and os.path.exists(own):
        ow = json.load(open(own))
        for s0 in ('REAL-agree', 'LONG'):
            if s0 in o:
                fl = S.flips(pr, s0, ow); o[s0]['flip_rate_vs_own_bf16'] = fl['rate']; o[s0]['n_vs_own'] = fl['n']
        for s_ in ('CF', 'CF-probe'):
            if s_ in o: o[s_]['fgh_vs_own_bf16'] = S.fgh_vs(s_, pr, ow)[0]
    res[f'{tag}::{cfg}::{sub}'] = o
    extra = ''
    s0 = 'REAL-agree' if 'REAL-agree' in o else 'LONG'
    if 'flip_rate_vs_own_bf16' in o.get(s0, {}):
        extra = f"\n   vs own bf16: {s0} flips {100*o[s0]['flip_rate_vs_own_bf16']:.2f}% (n {o[s0]['n_vs_own']})" + ''.join(f" {s_} fgh {o[s_]['fgh_vs_own_bf16']:.3f}" for s_ in ('CF', 'CF-probe') if s_ in o)
    print(f'### {tag} :: {cfg} :: {sub} (items {len(pr)})\n   ' + S.fmt(o) + extra, flush=True)
json.dump(res, open(os.path.expanduser('~/decider2/h3/scores_' + pat.replace('*', 'X') + '.json'), 'w'), indent=1)
