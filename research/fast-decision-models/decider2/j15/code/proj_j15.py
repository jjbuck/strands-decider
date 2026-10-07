"""Projection (arithmetic) of the measured A10G kernel splits to RTX 3090 / 4090 / 5090 with H2's model (h2/code/proj.py): GEMMs scale with
the card's dense peak at their precision, attention with the bf16 rate, glue/GDN/other with DRAM bandwidth; host time kept.
5090: W4A4 runs as NVFP4 (2x FP8/INT8 rate). bf16 GEMMs use fp16 accumulation on GeForce. Depth cascade = layer-cost x b8 on each card."""
import json, sys
sys.path.insert(0, '~/decider2/h2/code')
from proj import DEV, project
R = [json.loads(l) for l in open('~/decider2/j15/res/res_bench.jsonl')]
rec = {(r['prec'], r['T']): r for r in R if r['mode'] == 'full' and r.get('kern')}
pmap = {'b8': 'w8a8_b8', 'w4a4': 'w4a4', 'k48': 'k48', 'bf16': 'bf16'}
out = {}
for T in (256, 1000, 4000):
    row = {}
    for p in ('bf16', 'b8', 'w4a4', 'k48'):
        r = dict(rec[(p, T)]); r['prec'] = pmap[p]
        row[p] = {'A10G': r['wall']['median']}
        for dev in ('RTX3090', 'RTX4090', 'RTX5090'):
            row[p][dev] = round(project(r, dev, p == 'bf16')[0], 2)
    out[T] = row
    for dev in ('A10G', 'RTX3090', 'RTX4090', 'RTX5090'):
        b8 = row['b8'][dev]; w4 = row['w4a4'][dev]
        print(f"T={T} {dev:8s} bf16 {row['bf16'][dev]:7.2f} b8 {b8:7.2f} w4a4 {w4:7.2f} (D/V {w4/b8:.3f}) k48 {row['k48'][dev]:7.2f} | "
              f"W4A4 oracle-cascade >= {w4/b8 + 0.085:.3f}x | prec casc eps0 (f .65) {w4/b8 + 0.65:.2f}x | depth {{8,16}} {0.604*b8:6.2f} ms, {{16}} {0.694*b8:6.2f} ms")
json.dump(out, open('~/decider2/j15/res/proj.json', 'w'), indent=1)
