"""metrics.py: score every catalogued prediction set (and hobson + the evalkit baselines) on every suite.

    python3 metrics.py            # writes results/metrics.csv and results/metrics.json

Per suite: evalkit's own metrics (acc, agree, agree_sd, tv, CF flip / fgh, SHUF), plus
  brier = mean over decisions of sum over options (p - y)^2, as JevBench defines it (labelled suites only), and
  ece   = top-label expected calibration error, 10 equal-width bins.
Labelled questions: JB-all, JB-hard, REAL-label (Opus labels), CF and CF-probe items (labels by construction).
"""
import csv, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser('~/decider2/evalkit'))
import evalkit as EK

A = os.path.expanduser('~/decider2/analysis')
SUITES = ['JB-all', 'JB-hard', 'REAL-agree', 'LONG', 'CF', 'CF-probe', 'SHUF', 'REAL-label']
KEEP = {'acc', 'agree', 'agree_sd', 'tv', 'kl', 'coverage', 'flip', 'flip_given_hobson', 'dir', 'change', 'both_right', 'acc_consensus', 'n_q', 'n_pairs'}


def labelled(suite):
    """[(item_id, q, gold_label)]"""
    if suite in ('CF', 'CF-probe'):
        out = []
        for p in EK._pairs(suite): out += [(p['a'], p['q'], p['ea']), (p['b'], p['q'], p['eb'])]
        return out
    if suite in ('JB-all', 'JB-hard', 'REAL-label'):
        return [(it['id'], q, (it.get('expected') or {}).get(q)) for it in EK.load_suite(suite) for q in it['questions'] if (it.get('expected') or {}).get(q) is not None]
    return []


LAB = {s: labelled(s) for s in SUITES}


def calib(suite, preds):
    rows = []
    for iid, q, y in LAB[suite]:
        p = preds.get(iid, {}).get(q)
        if p is None: continue
        p = EK._norm(p)
        if y not in p: p = dict(p, **{y: 0.0})
        rows.append((p, y))
    if not rows: return {}
    brier = float(np.mean([sum((v - (k == y)) ** 2 for k, v in p.items()) for p, y in rows]))
    conf = np.array([max(p.values()) for p, _ in rows]); ok = np.array([max(p, key=p.get) == y for p, y in rows], float)
    bins = np.minimum((conf * 10).astype(int), 9); ece = 0.0
    for b in range(10):
        m = bins == b
        if m.any(): ece += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return dict(brier=brier, ece=float(ece), n_cal=len(rows))


def score_all(preds):
    out = {}
    for s in SUITES:
        m = EK.score(s, preds, baselines=False)['model']
        m = {k: v for k, v in m.items() if k in KEEP}
        m.update(calib(s, preds))
        out[s] = m
    return out


def refs_as_preds(which):
    """hobson or a baseline, merged across suites"""
    merged = {}
    for s in ('JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe'):
        merged.update(EK._as_preds(s, which))
    return merged


def main():
    os.makedirs(f'{A}/results', exist_ok=True)
    cat = list(csv.DictReader(open(f'{A}/catalog.csv')))
    res = {}
    for which in ['hobson'] + EK.SHOW:
        res[f'ref/{which}'] = score_all(refs_as_preds(which))
    for r in cat:
        safe = r['run'].replace('/', '__').replace(':', '~')
        preds = json.load(open(f'{A}/norm/{safe}.json'))
        res[r['run']] = score_all(preds)
    json.dump(res, open(f'{A}/results/metrics.json', 'w'), indent=1)
    cols = sorted({f'{s}.{k}' for v in res.values() for s, m in v.items() for k in m})
    with open(f'{A}/results/metrics.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['run'] + cols)
        for run, v in res.items():
            w.writerow([run] + [('' if (x := v.get(c.split('.')[0], {}).get(c.split('.', 1)[1])) is None or (isinstance(x, float) and math.isnan(x)) else round(x, 4)) for c in cols])
    print(f'{len(res)} runs scored -> {A}/results/metrics.csv')


if __name__ == '__main__':
    main()
