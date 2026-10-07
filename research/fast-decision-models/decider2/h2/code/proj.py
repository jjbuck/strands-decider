"""Project A10G-measured per-category kernel times to RTX 3090 / 4090 / 5090 (assumptions in DEV below; printed with the table).
Model: t_dev = gemm * R_gemm(prec) + attn * R_attn + (glue + gdn_core + other) * R_bw + host, R = A10G rate / device rate (same fraction of peak
as measured on the A10G). host = wall - busy on the A10G (H2D, launch of one graph, D2H), kept constant."""
import json, sys, os
# dense peak rates. A10G: effective peaks implied by the measured GEMM throughput (bf16 58 TF, int8 113 TOPS, int4 227 TOPS at ~81%):
# 70 TF bf16 (fp32 acc; fp16 acc gives no 2x on A10G, d1), 140 TOPS int8, 280 TOPS int4; 600 GB/s.
DEV = {
 'A10G': dict(bf16=70, bf16_f16acc=70, int8=140, int4=280, fp4=None, attn=70, bw=600),
 # GA102 whitepaper: FP16 tensor 142 TF with fp16 acc, 71 with fp32 acc; INT8 284; INT4 568 TOPS; 936 GB/s
 'RTX3090': dict(bf16=71, bf16_f16acc=142, int8=284, int4=568, fp4=None, attn=71, bw=936),
 # AD102 (Ada whitepaper): FP16 330 (fp16 acc) / 165 (fp32 acc); INT8 661; INT4 1321 TOPS; 1008 GB/s
 'RTX4090': dict(bf16=165, bf16_f16acc=330, int8=661, int4=1321, fp4=None, attn=165, bw=1008),
 # GB202: NVFP4 1676 dense (3352 "AI TOPS" is 2:4 sparse); FP8/INT8 838; FP16 419 (fp16 acc) / 209.5 (fp32 acc, GeForce half rate assumed); 1792 GB/s.
 # No INT4 tensor path on Blackwell: W4A4 runs as NVFP4 (16-element block scales: a different format than H1's per-token int4).
 'RTX5090': dict(bf16=209.5, bf16_f16acc=419, int8=838, int4=None, fp4=1676, attn=209.5, bw=1792),
}
def gemm_rate(d, prec, f16acc=False):
    if 'b8' in prec: prec = 'w8a8'          # W8A8 map with 8 bf16 GEMMs (~3% of GEMM time): scaled at the int8 ratio
    if 'k48' in prec or 'k24' in prec: prec = 'w4a4'   # int4/int8 mix: both scale by the same ratio on every device here
    if prec == 'bf16': return d['bf16_f16acc'] if f16acc else d['bf16']
    if prec in ('w8a8', 'w4a8'): return d['int8']
    if prec == 'w4a4': return d['int4'] if d['int4'] else d['fp4']
def project(rec, dev, f16acc=False):
    a = DEV['A10G']; d = DEV[dev]; k = rec['kern']; p = rec['prec']
    host = max(0.0, rec['wall']['median'] - k['busy'])
    g = k.get('gemm', 0) * gemm_rate(a, p, f16acc) / gemm_rate(d, p, f16acc)
    at = k.get('attn', 0) * a['attn'] / d['attn']
    mem = (k.get('glue', 0) + k.get('gdn_core', 0) + k.get('other', 0)) * a['bw'] / d['bw']
    return g + at + mem + host, dict(gemm=g, attn=at, mem=mem, host=host)
if __name__ == '__main__':
    f = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/decider2/h2/res_bench_v2.jsonl')
    R = [json.loads(l) for l in open(f)]
    print('| config | A10G measured | 3090 | 3090 bf16 fp16-acc | 4090 | 5090 |'); print('|---|---|---|---|---|---|')
    for r in R:
        row = f"| {r['prec']} T={r['T']} {r['bundle']} {r['mode']} | {r['wall']['median']:.1f} |"
        for dev in ('RTX3090',):
            row += f" {project(r, dev)[0]:.1f} |"
        row += f" {project(r, 'RTX3090', True)[0]:.1f} |" if r['prec'] == 'bf16' else ' - |'
        for dev in ('RTX4090', 'RTX5090'):
            row += f" {project(r, dev, r['prec'] == 'bf16')[0]:.1f} |"
        print(row)
