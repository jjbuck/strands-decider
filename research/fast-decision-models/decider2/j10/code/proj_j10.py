"""Project A10G-measured latencies (res_lat.jsonl, kernel split from one profiled graph replay) to RTX 3090 / 4090 / 5090 with H2's method and
device table (h2/code/proj.py): t = gemm * A10G_rate/dev_rate(prec) + attn * A10G_bf16/dev_bf16(fp16 acc on GeForce) + (glue+other) * A10G_bw/dev_bw + host.
Precisions: bf16 -> fp16-acc tensor rate on GeForce; i8 / w2 -> int8 rate; s4 -> int4 rate (Ampere/Ada) or NVFP4 rate (5090, ternary weights are
exact in E2M1; activations need block scales there); mix -> per-class split by FLOP share (qkv+gu = 65% of GEMM FLOPs at int4, o+down at int8).
python3 proj_j10.py res_lat.jsonl [res_hob_lat.jsonl]"""
import json, sys
DEV = {
    'A10G': dict(bf16=70, f16acc=70, int8=140, int4=280, fp4=None, bw=600),
    'RTX3090': dict(bf16=71, f16acc=142, int8=284, int4=568, fp4=None, bw=936),
    'RTX4090': dict(bf16=165, f16acc=330, int8=661, int4=1321, fp4=None, bw=1008),
    'RTX5090': dict(bf16=209.5, f16acc=419, int8=838, int4=None, fp4=1676, bw=1792),
}
FLOP_SHARE_INT4 = (6.55 + 1.64 + 1.64 + 17.7 + 17.7) / (6.55 + 1.64 + 1.64 + 6.55 + 17.7 * 3)   # qkv + gate/up share of BitNet GEMM FLOPs


def rate(d, prec):
    if prec == 'bf16': return d['f16acc']
    if prec in ('i8', 'w2', 'auto', 'c8'): return d['int8']
    if prec == 's4': return d['int4'] or d['fp4']
    raise ValueError(prec)


def gemm_scale(dev, prec):
    a = DEV['A10G']; d = DEV[dev]
    if prec in ('mix', 'mixw2', 'mixc'):
        f = FLOP_SHARE_INT4
        # time fractions on the A10G: int4 part runs at 2x the int8 rate
        t4 = f / a['int4']; t8 = (1 - f) / a['int8']
        return (t4 * a['int4'] / rate(d, 's4') + t8 * a['int8'] / rate(d, 'i8')) / (t4 + t8)
    return rate(a, prec) / rate(d, prec)


def project(rec, dev):
    k = rec['kern']; p = rec['prec']; a = DEV['A10G']; d = DEV[dev]
    host = max(0.0, rec['wall']['median'] - k['busy'])
    g = k.get('gemm', 0) * gemm_scale(dev, p)
    at = k.get('attn', 0) * a['f16acc'] / d['f16acc']
    mem = (k.get('glue', 0) + k.get('other', 0)) * a['bw'] / d['bw']
    return g + at + mem + host


if __name__ == '__main__':
    R = [json.loads(l) for l in open(sys.argv[1])]
    print('| prec | T | nq | A10G median (p95) | gemm / attn / glue+other ms | 3090 | 4090 | 5090 |'); print('|---|---|---|---|---|---|---|---|')
    for r in sorted(R, key=lambda r: (r['prec'], r['nq'], r['T'])):
        k = r['kern']
        print(f"| {r['prec']} | {r['T']} | {r['nq']} | {r['wall']['median']:.2f} ({r['wall']['p95']:.2f}) | {k.get('gemm',0):.2f} / {k.get('attn',0):.2f} / {k.get('glue',0)+k.get('other',0):.2f} | "
              + ' | '.join(f"{project(r, dv):.1f}" for dv in ('RTX3090', 'RTX4090', 'RTX5090')) + ' |')
