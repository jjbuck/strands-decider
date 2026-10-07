"""Laptop scorer: every config in preds files against hobson refs and the in-runtime bf16 ('dense'), checked against BRIEF10's bar.
python3 q1score.py --dense dense.json f1.json [f2.json ...]   -> prints + writes ~/decider2/q1/res/scores.json (merged by key file::tag)"""
import sys, os, json, math
sys.path[:0] = [os.path.expanduser('~/decider2/h1/code'), os.path.expanduser('~/decider2/evalkit')]
import h1score as HS

BAR = dict(real_flip=0.007, cf=0.99, cfp=0.95, real_label=0.78)


def check(o, meta=None):
    r = o.get('REAL-agree', {}); cf = o.get('CF', {}); cp = o.get('CF-probe', {}); jb = o.get('JB-hard', {}); rl = o.get('REAL-label', {})
    c = dict(real_flip=r.get('flip_rate'), real_flips=r.get('flips'), real_n=r.get('n'), tv=r.get('tv'),
             paired_vs_bf16=r.get('paired'), cf_fgh=cf.get('fgh'), cfp_fgh=cp.get('fgh'), jb_acc=jb.get('acc'), jb_mcnemar=(jb.get('mcnemar') or {}).get('p'),
             real_label=rl.get('acc'), long_flip=o.get('LONG', {}).get('flip_rate'))
    ok = dict(real_flip=c['real_flip'] is not None and c['real_flip'] <= BAR['real_flip'],
              paired=(c['paired_vs_bf16'] or {}).get('p', 0) > 0.05,
              cf=c['cf_fgh'] is not None and c['cf_fgh'] >= BAR['cf'],
              cfp=c['cfp_fgh'] is not None and c['cfp_fgh'] >= BAR['cfp'],
              jb=c['jb_mcnemar'] is not None and c['jb_mcnemar'] > 0.05,
              real_label=c['real_label'] is not None and c['real_label'] >= BAR['real_label'])
    c['pass'] = all(ok.values()); c['ok'] = ok
    if meta: c['work'] = meta.get('work'); c['spec'] = meta.get('spec')
    return c


if __name__ == '__main__':
    args = sys.argv[1:]; dense = None
    if '--dense' in args:
        j = args.index('--dense'); dense = json.load(open(args[j + 1]))['preds']['dense']; del args[j:j + 2]
    outp = os.path.expanduser('~/decider2/q1/res/scores.json')
    allres = json.load(open(outp)) if os.path.exists(outp) else {}
    for f in args:
        d = json.load(open(f)); P = d['preds']; M = d.get('meta', {})
        for t, pr in P.items():
            if not pr: continue
            o = HS.row(pr, dense if t != 'dense' else None)
            c = check(o, M.get(t))
            allres[f'{os.path.basename(f)}::{t}'] = dict(full=o, bar=c)
            w = (c.get('work') or {}).get('s', {})
            print(f"### {os.path.basename(f)}::{t}  n_items {len(pr)}  4-bit share (state rows) {w.get('int4', float('nan')):.3f} extra {w.get('extra_bf16_macs', 0):.3f}\n   "
                  f"REAL flips {c['real_flips']}/{c['real_n']} = {100*(c['real_flip'] or float('nan')):.2f}% (bar <= 0.70%) | paired vs bf16 {c['paired_vs_bf16']} | TV {c['tv'] if c['tv'] is None else round(c['tv'],4)}\n   "
                  f"CF fgh {c['cf_fgh']} (>= .99) | CF-probe fgh {c['cfp_fgh']} (>= .95) | JB-hard {c['jb_acc']} McNemar p {c['jb_mcnemar']} | REAL-label {c['real_label']} (>= .78) | LONG flips {c['long_flip']}\n   "
                  f"PASS {c['pass']} {c['ok']}", flush=True)
    json.dump(allres, open(outp, 'w'), indent=1)
