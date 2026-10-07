"""J7: latency on real traffic (arithmetic from measured anchors).  For each REAL-agree / LONG item: Qwen state + question tokens (hobson) vs
super-token state + question tokens (from an eval run's .ntok.json), each mapped through the MEASURED fused-runtime curve (lat_j7.json,
1-question configs, piecewise-linear in state rows between the exact lengths; question rows priced at the measured Q4-Q1 slope).
python3 traffic_lat.py lat_j7.json T64k.json.ntok.json OUT.json"""
import sys, json, numpy as np
sys.path.insert(0, '~/decider2/evalkit')
import evalkit as EK

lat = json.load(open(sys.argv[1])); nt = json.load(open(sys.argv[2])); QN = sys.argv[4] if len(sys.argv) > 4 else 'Q1'


def curve(cfg):
    xs, ys = [], []
    for k, v in lat.items():
        if k == 'meta': continue
        T, qn, c = k.split('_', 2)
        if qn == QN and c == cfg: xs.append(v['state_rows']); ys.append(v['median'])
    o = np.argsort(xs); return np.array(xs)[o], np.array(ys)[o]


def f(cfg):
    xs, ys = curve(cfg)
    def g(T):
        if T <= xs[-1]: return float(np.interp(T, xs, ys))
        return float(ys[-1] + (T - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]))
    return g


out = {}
for suite in ('REAL-agree', 'LONG'):
    ids = [it['id'] for it in EK.load_suite(suite)]
    rows = []
    gq = f('qwen'); gs = f('64k')
    for i in ids:
        if i not in nt: continue
        n = nt[i]
        # one question per request (the first), the request's own token counts
        Tq = n['s_orig'] + n['q_orig'][0]; Ts = n['s'] + n['q'][0]
        rows.append((Tq, Ts, gq(Tq), gs(Ts)))
    a = np.array(rows)
    out[suite] = dict(n=len(a), tok_median_qwen=float(np.median(a[:, 0])), tok_median_super=float(np.median(a[:, 1])),
                      ms_median_qwen=round(float(np.median(a[:, 2])), 2), ms_median_super=round(float(np.median(a[:, 3])), 2),
                      ms_p90_qwen=round(float(np.percentile(a[:, 2], 90)), 2), ms_p90_super=round(float(np.percentile(a[:, 3], 90)), 2),
                      speedup_median=round(float(np.median(a[:, 2] / a[:, 3])), 3), speedup_total=round(float(a[:, 2].sum() / a[:, 3].sum()), 3))
    print(suite, out[suite])
json.dump(out, open(sys.argv[3], 'w'), indent=1)
