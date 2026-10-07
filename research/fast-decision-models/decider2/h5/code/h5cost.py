"""GEMM-time model for a per-site precision map on the A10G, from H2's measured CUTLASS times (h2/res_gemm.json, M=1000 / 4000).
W4A4 -> s4 x s4; W4A8 -> s8 x s4 (H2's mixed-input kernel); W8A8 -> s8; bf16 -> cuBLAS. Wo shape = 'out' for both GDN and attention."""
import json, sys
G = json.load(open('~/decider2/h2/res_gemm.json'))
T = {}
for r in G:
    T[(r['shape'], r['M'])] = r
KEY = {(4, 4): 's4', (4, 8): 's8s4', (8, 8): 's8', (16, 16): 'bf16', (16, 4): 'bf16', (4, 16): 'bf16'}
GDN = [i for i in range(24) if i not in (3, 7, 11, 15, 19, 23)]


def gemm_ms(cfg, M=1000):
    tot = 0.0
    for i in range(24):
        g = i in GDN
        for site, shape in (('Win_g' if g else 'Win_a', 'gdn_in' if g else 'attn_in'), ('Wo_g' if g else 'Wo_a', 'out'), ('Wgu', 'gate_up'), ('Wd', 'down')):
            b = tuple(cfg.get(f'{i}.{site}', cfg[site]))
            tot += T[(shape, M)][KEY[b]]
    return tot / 1000


if __name__ == '__main__':
    CL = ('Win_g', 'Win_a', 'Wo_g', 'Wo_a', 'Wgu', 'Wd')
    for name, c in [('bf16', {k: (16, 16) for k in CL}), ('w8a8', {k: (8, 8) for k in CL}), ('w4a8', {k: (4, 8) for k in CL}), ('w4a4', {k: (4, 4) for k in CL})]:
        print(name, round(gemm_ms(c, 1000), 2), round(gemm_ms(c, 4000), 2))
    for k in CL:
        c = {x: (4, 4) for x in CL}; c[k] = (4, 8)
        print('w4a4 but', k, 'a8', round(gemm_ms(c, 1000), 2), round(gemm_ms(c, 4000), 2))
