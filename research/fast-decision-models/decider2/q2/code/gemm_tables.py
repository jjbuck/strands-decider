"""Laptop: sum standalone GEMM timings over hobson's 96 GEMMs per format (res_gemm_base / res_gemm_var / res_rr) -> ratios to W8A8-b8.
W8A8-b8 = 88 CUTLASS s8 GEMMs (gate_up via EVT SwiGLU) + 8 cuBLAS bf16 (7 Wo + 1 Wd), as deployed (H1 map b8)."""
import json, os
D = os.path.expanduser('~/decider2/q2/res')
CNT = {'gdn_in': 18, 'attn_in': 6, 'out': 24, 'gate_up': 24, 'down': 24}
base = {(r['M'], r['shape']): r for r in map(json.loads, open(f'{D}/res_gemm_base.jsonl'))}
var = {(r['M'], r['shape']): r for r in map(json.loads, open(f'{D}/res_gemm_var.jsonl'))}
out = {}
for M in (140, 400, 1125, 4125):
    b = {s: base[(M, s)] for s in CNT}; v = {s: var[(M, s)] for s in CNT}
    t = lambda d, s, k: d[s][k][1] if d[s].get(k) else None
    b8 = sum(CNT[s] * t(b, s, 'h2_s8') for s in ('gdn_in', 'attn_in')) + 17 * t(b, 'out', 'h2_s8') + 7 * t(b, 'out', 'bf16_cublas') \
        + 24 * t(b, 'gate_up', 'h2_s8_evt') + 23 * t(b, 'down', 'h2_s8') + t(b, 'down', 'bf16_cublas')
    w8all = sum(CNT[s] * (t(b, s, 'h2_s8_evt') if s == 'gate_up' else t(b, s, 'h2_s8')) for s in CNT)
    w4cut = sum(CNT[s] * (t(b, s, 'h2_s4_evt') if s == 'gate_up' else t(b, s, 'h2_s4')) for s in CNT)
    q2s4 = sum(CNT[s] * t(v, s, 'q2_s4') for s in CNT)
    row = dict(b8_ms=round(b8 / 1e3, 2), w8a8_all=round(w8all / b8, 3), w4a4_cutlass=round(w4cut / b8, 3), q2_s4=round(q2s4 / b8, 3))
    for k in v['gdn_in']:
        if k in ('kind', 'shape', 'M', 'N', 'K', 'q2_s4') or k.startswith('z_') or k.startswith('split_'): continue
        if all(v[s].get(k) for s in CNT):
            row[k] = round(sum(CNT[s] * t(v, s, k) for s in CNT) / b8, 3)
    # B5 / ResQ splits: keys differ by K (down has K = 6144); pair them by position in the list
    sk = [k for k in v['gdn_in'] if k.startswith('split_')]; skd = [k for k in v['down'] if k.startswith('split_')]
    for a, d in zip(sk, skd):
        row[a + ' / ' + d.replace('split_', '')] = round(sum(CNT[s] * t(v, s, (d if s == 'down' else a)) for s in CNT) / b8, 3)
    # B1 with the Z GEMM done separately by cuBLAS (as measured), per r
    for r in (32, 64, 128, 256):
        if all(v[s].get(f'b1_tbf16_r{r}') for s in CNT):
            row[f'b1_tbf16_r{r}+zcublas'] = round(sum(CNT[s] * (t(v, s, f'b1_tbf16_r{r}') + t(v, s, f'z_cublas_r{r}')) for s in CNT) / b8, 3)
    out[M] = row
    print(M, json.dumps(row))
json.dump(out, open(f'{D}/gemm_tables.json', 'w'), indent=1)
