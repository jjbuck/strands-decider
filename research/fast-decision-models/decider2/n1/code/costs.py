"""costs.py (laptop, pure python): latency + cost table from n1 results/lat_*.json, J8's method.
inf2.xlarge $0.76/h, both NeuronCores serving independent requests (2 requests per latency period);
A10G (g5.2xlarge) $1.212/h, one request at a time.  cost per million = price/3600 * latency_s / cores * 1e6."""
import json, glob, os
R = '~/decider2/n1/results'
INF2, A10G = 0.76, 1.212
A10G_MS = {64: 9.4, 256: 15.7, 1000: 52.7, 4000: 200.5}           # FAST_DECISION_MODEL.md / J8 [D]
J8_MS = {(64, 1): 30.9, (256, 1): 43.1, (1000, 1): 152.0, (4000, 1): 537.5, (64, 4): 94.8, (256, 4): 110.0, (1000, 4): 181.1}


def cost(ms, price, cores):
    return price / 3600 * ms / 1000 / cores * 1e6


rows = {}
for f in sorted(glob.glob(f'{R}/lat_*.json')):
    d = json.load(open(f))
    rows[d['tag']] = d
print(f"{'tag':28s} {'T':>5s} {'M':>2s} {'median':>8s} {'p95':>8s} {'J8':>7s} {'$/M inf2x2':>10s} {'$/M A10G':>9s}")
for tag, d in rows.items():
    T, M = d['T'], d['M']
    j8 = J8_MS.get((T, M))
    a = A10G_MS.get(T) if M == 1 else None
    print(f"{tag:28s} {T:5d} {M:2d} {d['median']:8.2f} {d['p95']:8.2f} {j8 if j8 else '-':>7} {cost(d['median'], INF2, 2):10.2f} "
          f"{cost(a, A10G, 1) if a else float('nan'):9.2f}")
