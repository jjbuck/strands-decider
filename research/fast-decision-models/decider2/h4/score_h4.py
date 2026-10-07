"""H4 scoring on the laptop (pure python): python3 score_h4.py preds/this-that-model-1.0_sf.jsonl [more.jsonl ...]"""
import sys, json, math, collections
sys.path.insert(0, '~/decider2/evalkit')
import numpy as np
import evalkit as EK


def load(p):
    preds = {}
    for l in open(p):
        r = json.loads(l)
        for q, d in r['p'].items():
            preds.setdefault(r['id'], {})[q] = d
    return preds


def arg(d): return max(d, key=d.get)


def mcnemar(a, b):
    """exact two-sided McNemar on paired booleans"""
    n01 = sum(1 for x, y in zip(a, b) if x and not y); n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0: return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return n01, n10, min(1.0, p)


def jb_family(preds):
    its = EK.load_suite('JB-hard'); hp = EK._as_preds('JB-hard', 'hobson')
    fam = collections.defaultdict(lambda: [0, 0, 0]); A = []; H = []
    for it in its:
        q = next(iter(it['questions'])); e = it['expected'][q]
        if it['id'] not in preds: continue
        a = arg(EK._norm(preds[it['id']][q])) == e; h = arg(EK._norm(hp[it['id']][q])) == e
        f = fam[it['family']]; f[0] += 1; f[1] += a; f[2] += h; A.append(a); H.append(h)
    return {k: (v[0], v[1] / v[0], v[2] / v[0]) for k, v in sorted(fam.items())}, mcnemar(A, H)


def main(paths):
    out = {}
    for p in paths:
        preds = load(p)
        print('\n' + '=' * 30, p, 'items', len(preds))
        EK.report(preds, suites=['JB-hard', 'JB-long', 'REAL-agree', 'LONG', 'CF', 'CF-probe', 'SHUF', 'REAL-label'], show=['merged_full', 'nostate', 'qattn10@7'])
        print()
        EK.kill_check(preds)
        fam, mc = jb_family(preds)
        print('\nJB-hard by family (n, this-that acc, hobson acc):')
        for k, v in fam.items(): print(f'  {k:28s} {v[0]:4d} {v[1]:.3f} {v[2]:.3f}')
        print('JB-hard McNemar (tt right & hobson wrong, hobson right & tt wrong, p):', mc)
        for s in ('CF', 'CF-probe'):
            for by in ('kind', 'domain', 'pos', 'len'):
                EK.print_breakdown(s, preds, by)
        out[p] = {s: EK.score(s, preds, baselines=False) for s in ['JB-hard', 'JB-long', 'JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe', 'SHUF', 'REAL-label']}
        out[p]['jb_family'] = fam; out[p]['jb_mcnemar'] = mc
        out[p]['cf_kind'] = EK.breakdown('CF', preds, 'kind'); out[p]['cfp_kind'] = EK.breakdown('CF-probe', preds, 'kind')
    json.dump(out, open('~/decider2/h4/scores.json', 'w'), indent=1, default=str)


if __name__ == '__main__':
    main(sys.argv[1:])
