"""J5 projections (laptop, arithmetic only) from measured A10G kernel splits (res_sbench*.jsonl) to RTX 3090 / 4090 / 5090.
GEMM: additive roofline split at the A10G, Tc = rows*2.745 GFLOP / R, Tm = weight bytes / BW (achievable 510 GB/s measured), f = Tc/(Tc+Tm);
      card time = g * (f * R_a10g/R_card + (1-f) * BW_a10g/BW_card)  (keeps the A10G's measured inefficiency; no extra overlap assumed).
non-GEMM (GDN, glue, attention, misc: short latency-bound kernels): n / s_card, s = 1.2 (3090), 1.5 (4090), 1.8 (5090)  [speculation].
Floor: the GEMM roofline max(Tc, Tm) at 85% of peak on that card, i.e. what a perfect megakernel could approach (non-GEMM ~0)."""
import json, sys, glob, os
FL = 2.745e9; P_ALL = 1.3726e9
# (int8 TOPS, fp16-MMA TFLOPS used for bf16/fp16 weights-path, bandwidth GB/s achievable ~85% of spec, non-GEMM speedup)
CARDS = {'A10G': dict(i8=140e12, f16=70e12, bw=510e9, s=1.0),
         '3090': dict(i8=284e12, f16=142e12, bw=0.85 * 936e9, s=1.2),     # fp16 accumulation for the bf16 path (GeForce 2x)
         '4090': dict(i8=660e12, f16=330e12, bw=0.85 * 1008e9, s=1.5),
         '5090': dict(i8=838e12, f16=419e12, bw=0.85 * 1792e9, s=1.8)}
BYTES = {'bf16': 2.0, 'w8a8': 1.0 + 0.03, 'w4a16': 0.53, 'w8a16': 1.0, 'w4a4': 0.5}
RATE = {'bf16': 'f16', 'w8a8': 'i8', 'w4a16': 'f16', 'w8a16': 'f16', 'w4a4': 'i4'}


def rate(card, fmt):
    c = CARDS[card]
    if RATE[fmt] == 'i4': return 2 * c['i8']
    return c[RATE[fmt]]


def project(rec, fmt):
    k = rec['kern']; rows = rec['rows']
    g = k.get('gemm', 0.0); n = rec['wall']['median'] - g
    a = CARDS['A10G']
    Tc = rows * FL / rate('A10G', fmt); Tm = P_ALL * BYTES[fmt] / a['bw']; f = Tc / (Tc + Tm)
    out = {}
    for c, spec in CARDS.items():
        gc = g * (f * rate('A10G', fmt) / rate(c, fmt) + (1 - f) * a['bw'] / spec['bw'])
        floor = 1000 * max(rows * FL / (0.85 * rate(c, fmt)), P_ALL * BYTES[fmt] / spec['bw'])
        out[c] = dict(proj=round(gc + n / spec['s'], 2), floor=round(floor, 2))
    return out


if __name__ == '__main__':
    files = sys.argv[1:] or sorted(glob.glob(os.path.expanduser('~/decider2/j5/res_sbench*.jsonl')))
    recs = {}
    for f in files:
        for l in open(f):
            r = json.loads(l); recs[r['key']] = r
    for key, r in sorted(recs.items(), key=lambda kv: (kv[1]['prec'], kv[1]['bundle'], kv[1]['T'])):
        p = r['prec']
        fmt = 'w8a8' if 'w8a8' in p else ('w4a4' if 'w4a4' in p else ('w4a16' if 'w4a16' in p else ('w8a16' if 'w8a16' in p else 'bf16')))
        pr = project(r, fmt)
        print(f"{p:40s} T={r['T']:4d} {r['bundle']} rows={r['rows']:4d} A10G {r['wall']['median']:6.2f} (gemm {r['kern'].get('gemm',0):5.2f}) | " +
              ' '.join(f"{c} {v['proj']:5.2f} [{v['floor']:4.2f}]" for c, v in pr.items() if c != 'A10G'))
