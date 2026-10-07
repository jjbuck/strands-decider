"""Projection (arithmetic) of Q2's measured A10G e2e kernel splits to RTX 3090 / 4090 / 5090, with H2's model (h2/code/proj.py):
GEMMs scale with the card's dense peak at their precision, attention with the bf16 rate, glue / GDN / other with DRAM bandwidth, host
time kept. Row-role (w4q8c, sequential record): the CUTLASS int4 state-row GEMMs scale with the int4 (5090: NVFP4) rate; the q2gemm
int8 question-row GEMMs (125 rows, below N* on every card) are weight-streaming bound, so they scale with DRAM bandwidth."""
import json, sys, os
sys.path.insert(0, os.path.expanduser('~/decider2/h2/code'))
from proj import DEV, project
rec = {}
for fn in ('res_e2e.jsonl', 'res_e2e_kern.jsonl', 'res_e2e_k64.jsonl'):
    fp = os.path.expanduser(f'~/decider2/q2/res/{fn}')
    if os.path.exists(fp):
        for l in open(fp):
            r = json.loads(l); rec[(r['name'], r['T'], r['bundle'])] = r
out = {}
for T in (64, 256, 1000, 4000):
    for bn in ('1q', '15q'):
        row = {}
        for name, prec in (('b8', 'w8a8_b8'), ('w4a4', 'w4a4')):
            r = dict(rec[(name, T, bn)]); r['prec'] = prec
            row[name] = {'A10G': r['wall']['median'], **{d: round(project(r, d)[0], 2) for d in ('RTX3090', 'RTX4090', 'RTX5090')}}
        if ('w4q8c', T, bn) in rec and 'gemm_q2' in rec[('w4q8c', T, bn)]['kern']:
            r = rec[('w4q8c', T, bn)]; k = r['kern']; a = DEV['A10G']
            host = max(0.0, r['wall']['median'] - k['busy'])
            row['w4q8c'] = {'A10G': r['wall']['median']}
            for d in ('RTX3090', 'RTX4090', 'RTX5090'):
                dv = DEV[d]
                g4 = k.get('gemm_cut', 0) * a['int4'] / (dv['int4'] or dv['fp4'])
                nq = r['rows'] - r['T']
                # 125 question rows sit below N* on every card (weight-streaming bound: bandwidth ratio); 3720 rows are compute bound (int8 ratio)
                g8 = k.get('gemm_q2', 0) * (a['bw'] / dv['bw'] if nq < 500 else a['int8'] / dv['int8'])
                at = k.get('attn', 0) * a['attn'] / dv['attn']
                mem = (k.get('glue', 0) + k.get('gdn_core', 0) + k.get('other', 0)) * a['bw'] / dv['bw']
                row['w4q8c'][d] = round(g4 + g8 + at + mem + host, 2)
        for nm in [n for (n, t, b) in rec if t == T and b == bn and 'k64rr' in n]:
            r = dict(rec[(nm, T, bn)]); r['prec'] = 'w8a8'      # 66% of its MACs are int8: all GEMM time scaled at the int8 ratio (conservative)
            row['k64rr'] = {'A10G': r['wall']['median'], **{d: round(project(r, d)[0], 2) for d in ('RTX3090', 'RTX4090', 'RTX5090')}}
        out[f'{T}|{bn}'] = row
        print(T, bn, json.dumps(row))
json.dump(out, open(os.path.expanduser('~/decider2/q2/res/proj.json'), 'w'), indent=1)
