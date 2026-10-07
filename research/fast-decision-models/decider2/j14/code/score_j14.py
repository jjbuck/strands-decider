"""laptop: python3 score_j14.py preds.jsonl [variant ...] -> compact table per variant (+ per-question breakdown on REAL/LONG sd misses)"""
import sys, os, json, collections
sys.path[:0] = [os.path.expanduser('~/decider2/j14/code'), os.path.expanduser('~/decider2/evalkit')]
import numpy as np
import evalkit as EK
import fscore as FS

def load_variants(f):
    P = collections.defaultdict(lambda: collections.defaultdict(dict)); meta = collections.defaultdict(list)
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        P[r['v']][r['id']][r['q']] = r['p']; meta[r['v']].append((r.get('nl'), r.get('Lq'), r.get('T')))
    return P, meta

def perq(name, preds):
    rows = [r for r in EK._qrows(name, preds) if r['pred'] is not None and r['hob'] is not None and r['nost'] is not None and EK._arg(r['nost']) != EK._arg(r['hob'])]
    c = collections.defaultdict(lambda: [0, 0])
    for r in rows: c[r['q']][0] += EK._arg(r['pred']) == EK._arg(r['hob']); c[r['q']][1] += 1
    return {k: (v[0] / v[1], v[1]) for k, v in sorted(c.items(), key=lambda kv: -kv[1][1])}

def sd_breakdown(name, preds):
    rows = [r for r in EK._qrows(name, preds) if r['pred'] is not None and r['hob'] is not None and r['nost'] is not None and EK._arg(r['nost']) != EK._arg(r['hob'])]
    c = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        k = 'proc' if 'procedure' in r['q'] else 'other'; c[k][0] += EK._arg(r['pred']) == EK._arg(r['hob']); c[k][1] += 1
    return {k: (v[0] / v[1], v[1]) for k, v in c.items()}

if __name__ == '__main__':
    f = sys.argv[1]; P, meta = load_variants(f)
    want = sys.argv[2:] or sorted(P)
    out = {}
    for v in want:
        p = {k: dict(x) for k, x in P[v].items()}
        r = FS.score_all(p); out[v] = r
        nl = [x[0] for x in meta[v] if x[0]]; lq = [x[1] for x in meta[v] if x[1]]
        def g(k, f2): return r.get(k, {}) and r[k].get(f2)
        line = [f'{v:<22} nq={len(meta[v])} live/q={np.mean(nl):.1f}/{np.mean(lq):.0f}']
        for k, f2 in (('REAL-agree agree', 'agree'), ('REAL-agree agree', 'agree_sd'), ('REAL-agree agree', 'tv'), ('LONG agree', 'agree'), ('LONG agree', 'agree_sd'),
                      ('CF pairs', 'pair_acc'), ('CF pairs', 'fgh'), ('CF-probe pairs', 'pair_acc'), ('CF-probe pairs', 'fgh'), ('JB-all', 'acc'), ('JB-hard', 'acc'), ('JB-hard', 'p'), ('REAL-label', 'acc'), ('REAL-label', 'p')):
            x = g(k, f2)
            if x is not None and x != {}: line.append(f'{k.split()[0]}.{f2}={x:.3f}')
        print(' '.join(line))
        bd = {s: sd_breakdown(s, p) for s in ('REAL-agree', 'LONG')}
        print('     sd by q:', {s: {k: f'{a:.3f}/{n}' for k, (a, n) in d.items()} for s, d in bd.items()})
        if os.environ.get('PERQ'):
            for s in ('REAL-agree', 'LONG'):
                print('     perq', s, ' '.join(f'{k}={a:.2f}/{n}' for k, (a, n) in perq(s, p).items() if n >= 5))
    json.dump(out, open(f.replace('.jsonl', '_scores.json'), 'w'), indent=1)
