"""J3 cost model (pure arithmetic, no torch). FLOPs per state token and latency projections for hobson and the depth-split
decision transformer (DT). Calibrated on the measured A10G anchors of hobson in the fused lean2 runtime (FAST_DECISION_MODEL 2.1):
GEMMs 46.0 ms at T=1000 (59.6 TFLOPS), non-GEMM 6.7 ms; T = 64/256/512/1000/4000 -> 9.4/15.7/28.3/52.7/200.5 ms.
Projection method (FAST_DECISION_MODEL sec 9): GEMM time / card GEMM ratio, memory-bound time / bandwidth ratio.
python3 costmodel.py"""
import json, sys
D, I = 2048, 6144
GDN_P = D * 8224 + D * D + 3 * D * I          # in_proj (qkv 6144 + z 2048 + b 16 + a 16), out_proj, MLP
ATT_P = D * 5120 + D * D + 3 * D * I          # q(+gate) 4096 + k 512 + v 512, o_proj, MLP
TYPES = ['g', 'g', 'g', 'a'] * 6              # attention at 3, 7, 11, 15, 19, 23
MEM_ATT = 2 * D * 1024                        # memory K/V of one deep attention layer, FLOP/2 (MACs) per state row
MEM_GDN = D * 4128                            # memory k, v, b, a of one deep GDN layer (q and z not needed for state rows)
TF = 59.6e12; BW = 600e9 * 0.80; ATT_TF = 40e12
NG_GDN = 0.256e-6; NG_OTH = 0.054e-6           # s per row: GDN kernel + conv per GDN layer; norms etc. per layer (measured split at T=1000)
OVH = 1.0e-3
CARDS = {'A10G': (1.0, 1.0), '3090': (2.0, 1.56), '4090': (4.74, 1.68), '5090': (5.93, 2.99)}   # (GEMM ratio, bandwidth ratio)


def lp(t): return GDN_P if t == 'g' else ATT_P


def flops_per_state_token(Ls=24, bridge='full'):
    """2 * MACs per state row. bridge 'A' = deep attention layers read memory K/V only; 'G' = also deep GDN layers (SwiftKV-style)."""
    f = sum(2 * lp(t) for t in TYPES[:Ls])
    for t in TYPES[Ls:]:
        if t == 'a': f += 2 * MEM_ATT
        elif bridge == 'G': f += 2 * MEM_GDN
    return f


def gemm(N, P):          # (compute part, weight-stream floor part) seconds
    return 2 * N * P / TF, 2 * P / BW


def latency(T, q=125, nq=1, Ls=24, bridge='A', card='A10G'):
    """T state rows, nq questions of q tokens each (packed; each question attends to state + itself). Returns ms."""
    gr, br = CARDS[card]
    Q = q * nq; N = T + Q
    tg = tm = 0.0                                      # GEMM-compute-bound seconds, memory-bound seconds
    for i, t in enumerate(TYPES):
        P = lp(t)
        rows = N if i < Ls else Q
        c, w = gemm(rows, P)
        if c >= w * gr / br: tg += c                  # compute-bound on this card
        else: tm += w
        tm += rows * (NG_GDN if t == 'g' else 0.0) + rows * NG_OTH
        if t == 'a':                                    # attention core (memory-ish flash; scale with bandwidth)
            keys = N
            tm += (4 * 8 * 256 * (rows * keys if i >= Ls else N * N / 2)) / ATT_TF
        if i >= Ls:                                     # memory path on the T state rows
            if t == 'a': tg += 2 * T * MEM_ATT / TF
            elif bridge == 'G':
                tg += 2 * T * MEM_GDN / TF; tm += T * NG_GDN
    return 1000 * (tg / gr + tm / br + OVH / br)


if __name__ == '__main__':
    out = {}
    print('FLOPs per state token (GFLOP):')
    rows = [('hobson', 24, 'A')] + [(f'DT-{b}{L}', L, b) for b in 'AG' for L in (4, 8, 12)]
    for nm, L, b in rows:
        f = flops_per_state_token(L, b); out[f'flops_{nm}'] = f
        print(f'  {nm:10s} {f / 1e9:6.3f}  ({f / flops_per_state_token():.3f}x)')
    print('per question row: %.3f GFLOP (all 24 layers, same as hobson)' % (flops_per_state_token() / 1e9))
    print('\nmodel check vs measured hobson A10G (1 question folded into T, q=0):', {T: round(latency(T, q=0), 1) for T in (64, 256, 512, 1000, 4000)},
          'measured', {64: 9.4, 256: 15.7, 512: 28.3, 1000: 52.7, 4000: 200.5})
    Ts = (64, 128, 256, 400, 1000, 4000)
    for card in CARDS:
        for nq in (1, 4):
            print(f'\n{card}, {nq} question(s) of 125 tokens (ms):')
            print('  %-10s' % 'model' + ''.join('%8d' % T for T in Ts))
            for nm, L, b in rows:
                v = [latency(T, 125, nq, L, b, card) for T in Ts]; out[f'lat_{card}_q{nq}_{nm}'] = dict(zip(Ts, v))
                print('  %-10s' % nm + ''.join('%8.1f' % x for x in v))
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else '/dev/null', 'w'), indent=1)
