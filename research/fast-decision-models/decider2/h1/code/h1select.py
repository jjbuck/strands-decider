"""H1 mixed-precision selection from a single-GEMM sensitivity file (error propagation to the decision, additive at 4 bits).
GEMM time model: A10G CUTLASS, M=1024 (G2 res_prec.json), us per GEMM: bf16 / s8 / s4.
python3 h1select.py sens_w4g.json [--hi w8a8] -> prints the KL-vs-GEMM-time frontier and writes precmap_k{N}.json for several N"""
import sys, json
T = {('gdn', 'Win'): (676, 287, 161), ('att', 'Win'): (349, 186, 109), ('*', 'Wo'): (183, 105, 67), ('*', 'Wgu'): (843, 423, 228), ('*', 'Wd'): (428, 261, 145)}
ATT = (3, 7, 11, 15, 19, 23)
def t(i, k, p):
    key = (('att' if i in ATT else 'gdn'), k) if k == 'Win' else ('*', k)
    return T[key][{'bf16': 0, 'w8a8': 1, 'w4a4': 2}[p]]
d = json.load(open(sys.argv[1])); acc = d['acc']
keys = [c for c in acc if c != 'ALL']
base_t = sum(t(int(c.split('.')[0]), c.split('.')[1], 'w4a4') for c in keys) / 1000
tot = sum(acc[c]['kl'] for c in keys)
print(f'all w4a4: GEMM {base_t:.1f} ms/1k tok, measured joint KL {acc["ALL"]["kl"]:.4f}, sum singles {tot:.4f}')
# greedy by KL removed per extra microsecond
order = sorted(keys, key=lambda c: -acc[c]['kl'] / (t(int(c.split('.')[0]), c.split('.')[1], 'w8a8') - t(int(c.split('.')[0]), c.split('.')[1], 'w4a4')))
cum_t = base_t; cum_kl = tot; out = []
for n, c in enumerate(order, 1):
    i, k = int(c.split('.')[0]), c.split('.')[1]
    cum_t += (t(i, k, 'w8a8') - t(i, k, 'w4a4')) / 1000; cum_kl -= acc[c]['kl']
    out.append((n, c, round(cum_t, 2), round(cum_kl / tot, 3)))
for r in out:
    if r[0] in (4, 8, 12, 16, 24, 32, 48, 64, 96) or r[0] <= 3: print(r)
json.dump(dict(rank=order, src=sys.argv[1]), open(sys.argv[1].replace('.json', '_rank_eff.json'), 'w'))
