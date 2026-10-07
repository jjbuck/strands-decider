"""laptop: lat.json -> markdown latency table (median / p95 ms) + deltas vs hobson + projections [arithmetic]."""
import json, sys
L = json.load(open(sys.argv[1]))
TS = [64, 128, 256, 400, 1000, 4000]
names = ['hob', 'bi_all', 'bi_all_nc', 'bi_early', 'qa_all']
for M in (1, 4):
    print(f'\n**{M} question{"s" if M > 1 else ""}** (median ms, p95 in brackets; Δ vs hob)\n')
    print('| config | ' + ' | '.join(str(t) for t in TS) + ' |'); print('|---' * (len(TS) + 1) + '|')
    for n in names:
        cells = []
        for t in TS:
            r = L.get(f'{n}|T{t}|M{M}'); h = L.get(f'hob|T{t}|M{M}')
            if not r: cells.append('-'); continue
            c = f"{r['median']:.1f} [{r['p95']:.1f}]"
            if n != 'hob' and h: c += f" ({r['median'] - h['median']:+.1f})"
            cells.append(c)
        print(f'| {n} | ' + ' | '.join(cells) + ' |')
# projections (compute-bound part scales like hobson's documented projection; the bidirectional extra (GDN scans, flips, attention) by DRAM bandwidth)
R = {'3090': 27.4 / 52.7, '4090': 13.7 / 52.7, '5090': 10.0 / 52.7}
BW = {'3090': 600 / 936, '4090': 600 / 1008, '5090': 600 / 1792}
print('\n**Projections, 1 question [arithmetic]** (hobson part x documented card ratio at 1000 tokens; extra x bandwidth ratio)\n')
print('| T | config | A10G | 3090 | 4090 | 5090 |'); print('|---|---|---|---|---|---|')
for t in (1000, 4000):
    h = L.get(f'hob|T{t}|M1')
    for n in ('hob', 'bi_all', 'bi_all_nc', 'qa_all'):
        r = L.get(f'{n}|T{t}|M1')
        if not r or not h: continue
        d = r['median'] - h['median']
        print(f"| {t} | {n} | {r['median']:.1f} | " + ' | '.join(f"{h['median'] * R[c] + d * BW[c]:.1f}" for c in R) + ' |')
