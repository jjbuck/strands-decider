"""latency model from the measured grid (res_bench.jsonl): per-question latency at its exact row count by linear interpolation in rows."""
import json, numpy as np, collections
RB = '~/decider2/j15/res/res_bench.jsonl'


def curves(path=RB):
    c = collections.defaultdict(dict)
    for l in open(path):
        r = json.loads(l); c[(r['prec'], r['mode'])][r['rows']] = r['wall']['median']
    return {k: (np.array(sorted(v)), np.array([v[x] for x in sorted(v)])) for k, v in c.items()}


def at(cv, rows):
    x, y = cv
    rows = np.asarray(rows, dtype=float)
    out = np.interp(rows, x, y)
    hi = rows > x[-1]
    if hi.any():
        s = (y[-1] - y[-2]) / (x[-1] - x[-2]); out[hi] = y[-1] + s * (rows[hi] - x[-1])
    return out


def summ(v):
    v = np.asarray(v)
    return dict(mean=float(v.mean()), p50=float(np.median(v)), p95=float(np.quantile(v, 0.95)), n=int(len(v)))
