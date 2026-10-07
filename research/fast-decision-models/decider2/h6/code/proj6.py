"""H6 projections from the A10G latency matrix (H2's method, ~/decider2/h2/code/proj.py): GEMM time scales with the device's dense peak for
the precision, attention with the bf16 peak, memory-bound kernels with DRAM bandwidth, host time constant.  Mixed-row configs: the extra time
of the bf16 rows (schemamix - schema, plainqb - plain, same T / bundle / overlap setting) is weight-read-bound skinny bf16 GEMMs, so it is
scaled with DRAM bandwidth instead of the int4 rate.  5090: W4A4 -> NVFP4 rate (different format; speculation).
python3 proj6.py res_bench_*.jsonl"""
import sys, json, os
sys.path.insert(0, os.path.expanduser('~/decider2/h2/code'))
from proj import DEV, project
R = []
for f in sys.argv[1:]:
    R += [json.loads(l) for l in open(f)]
by = {(r['key'].split('|')[0], r['T'], r['bundle'], r['mode']): r for r in R}


def prec_of(tag):
    return 'bf16' if tag.startswith('bf16') else ('k48' if 'k48' in tag else 'w4a4')


rows = []
for (tag, T, bn, md), r in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
    r = dict(r); r['prec'] = prec_of(tag)
    base = None
    if md in ('schemamix', 'schemapq'): base = by.get((tag.replace('ovl', ''), T, bn, 'schema')) or by.get((tag, T, bn, 'schema'))
    if md == 'plainqb': base = by.get((tag.replace('ovl', ''), T, bn, 'plain')) or by.get((tag, T, bn, 'plain'))
    out = {}
    for dev in ('RTX3090', 'RTX4090', 'RTX5090'):
        f16 = r['prec'] == 'bf16'
        if base is None:
            out[dev] = project(r, dev, f16)[0]
        else:
            b = dict(base); b['prec'] = prec_of(tag)
            extra = max(0.0, r['wall']['median'] - b['wall']['median'])
            out[dev] = project(b, dev, f16)[0] + extra * DEV['A10G']['bw'] / DEV[dev]['bw']
    rows.append((tag, T, bn, md, r['wall']['median'], r['wall']['p95'], out))
print('| config | T | Q | layout | A10G median | A10G p95 | 3090 | 4090 | 5090 |'); print('|---|---|---|---|---|---|---|---|---|')
for tag, T, bn, md, med, p95, out in rows:
    print(f"| {tag} | {T} | {bn} | {md} | {med:.1f} | {p95:.1f} | {out['RTX3090']:.1f} | {out['RTX4090']:.1f} | {out['RTX5090']:.1f} |")
