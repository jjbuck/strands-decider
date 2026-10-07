import json, sys, os
sys.path.insert(0, os.path.expanduser('~/decider2/h2/code'))
import proj as PJ
R = [json.loads(l) for l in open(os.path.expanduser('~/decider2/h2/res_bench_v4.jsonl'))]
d = {r['key']: r for r in R}
P = ['bf16', 'w8a8', 'w4a8', 'w4a4', 'precmap_w8a8_b8:w8a8', 'precmap_w4a4_k48']
print('| T | questions | layout | rows computed | bf16 | W8A8 | W4A8 | W4A4 | W8A8-GPTQ b8 | W4A4 k48 |'); print('|---|---|---|---|---|---|---|---|---|---|')
for T in (1000, 4000):
    for bn in ('1q', '4q', '15q'):
        for md in ('plain', 'schema'):
            r0 = d[f'bf16|{T}|{bn}|{md}']
            print(f"| {T} | {bn} | {md} | {r0['rows']}{' (+P '+str(r0['P'])+')' if md == 'schema' else ''} | " + ' | '.join(f"{d[f'{p}|{T}|{bn}|{md}']['wall']['median']:.1f}" for p in P) + ' |')
print('max p95-median', max(r['wall']['p95'] - r['wall']['median'] for r in R))
print('\nbreakdown')
for T in (1000, 4000):
    for bn, md in (('1q', 'schema'), ('15q', 'schema')):
        for p in ('bf16', 'w8a8', 'w4a4', 'precmap_w8a8_b8:w8a8'):
            r = d[f'{p}|{T}|{bn}|{md}']; k = r['kern']
            print(f"| {T} {bn} {md} | {p} | {r['wall']['median']:.1f} | {k.get('gemm',0):.1f} | {k.get('attn',0):.1f} | {k.get('glue',0):.1f} | {k.get('gdn_core',0):.1f} | {k.get('other',0):.2f} | {r['wall']['median']-k['busy']:.2f} | {100*(1-k.get('gemm',0)/r['wall']['median']):.0f}% |")
print('\nprojections')
for T in (1000, 4000):
    for bn, md in (('1q', 'schema'), ('15q', 'schema'), ('15q', 'plain')):
        for p in ('bf16', 'w8a8', 'w4a4', 'precmap_w8a8_b8:w8a8'):
            r = d[f'{p}|{T}|{bn}|{md}']; f16 = p == 'bf16'
            print(f"| {T} {bn} {md} | {p} | {r['wall']['median']:.1f} | {PJ.project(r, 'RTX3090', f16)[0]:.1f} | {PJ.project(r, 'RTX4090', f16)[0]:.1f} | {PJ.project(r, 'RTX5090', f16)[0]:.1f} |")
r = d['w4a4|1000|1q|schema']
for dev in ('RTX3090', 'RTX4090', 'RTX5090'):
    print(dev, {k: round(v, 2) for k, v in PJ.project(r, dev)[1].items()})
