"""M1: RuleTaker / holdout probe scoring (laptop). python3 score_holdout.py holdout.json [more.json ...] -> per task: n, hobson, model, McNemar p"""
import sys, json, math


def mcnemar(rows):
    b = sum(1 for r in rows if r[1] and not r[0]); c = sum(1 for r in rows if r[0] and not r[1]); n = b + c
    if n == 0: return 1.0, b, c
    k = min(b, c); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), b, c


for f in sys.argv[1:]:
    d = json.load(open(f)); print(f)
    for t, r in sorted(d.items()):
        p, b, c = mcnemar(r['rows'])
        print(f"  {t:18s} n {r['n']:4d}  hobson {r['hobson']:.3f}  model {r['model']:.3f}  agree {r['agree']:.3f}  model-only/hobson-only {b}/{c}  McNemar p {p:.3g}")
