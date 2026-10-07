"""M1 projections (laptop, arithmetic) from the measured A10G wall times and kernel splits (FAST_DECISION_MODEL sec 9 method, J3/J14 ratios).
  card time = GEMM_A10G * gemm_factor + attention_A10G / attn_ratio + (GDN + conv/prep + other)_A10G / bw_ratio + (wall - kernel sum)_A10G
  gemm_factor: J14's additive roofline split at the A10G (compute part / fp16-accumulation tensor ratio, weight-stream part / bandwidth ratio).
  attention: FlashAttention accumulates in fp32, which runs at half the fp16-accumulation rate on GeForce cards, so attn_ratio = half the GEMM ratio.
python3 proj_m1.py PROF.json LAT.json [LAT2.json ...] > table"""
import json, sys
CARDS = {'3090': dict(g=2.0, a=1.0, bw=936 / 600), '4090': dict(g=4.74, a=2.37, bw=1008 / 600), '5090': dict(g=5.93, a=2.97, bw=1792 / 600)}
P_ALL = 1.3726e9; FL = 2.745e9; R_A10G = 70e12; BW_A10G = 600e9 * 0.85
prof = json.load(open(sys.argv[1])); lat = {}
for f in sys.argv[2:]:
    for k, v in json.load(open(f)).items():
        if k != 'meta': lat[k] = v
QT = {'S': 93, 'B': 364}
out = {}
for key, pr in sorted(prof.items()):
    kind, T = key.split('_T'); T = int(T)
    lk = {'hobS': f'Q1_T{T}_hobS', 'm1S': f'Q1_T{T}_m1S', 'hobB': f'Q4_T{T}_hobB', 'm1B': f'Q4_T{T}_m1B'}[kind]
    if lk not in lat: continue
    wall = lat[lk]['median']; rows = T + QT[kind[-1]]
    g = pr.get('gemm', 0); a = pr.get('attention', 0); mem = pr.get('gdn', 0) + pr.get('conv_prep', 0) + pr.get('other', 0)
    over = max(0.0, wall - pr['total'])
    Tc = rows * FL / R_A10G; Tm = P_ALL * 2 / BW_A10G; f = Tc / (Tc + Tm)
    row = dict(A10G=wall, gemm=round(g, 2), attn=round(a, 2), gdn=round(pr.get('gdn', 0), 2), mem_other=round(mem, 2), overhead=round(over, 2))
    for c, r in CARDS.items():
        row[c] = round(g * (f / r['g'] + (1 - f) / r['bw']) + a / r['a'] + mem / r['bw'] + over, 1)
    out[lk] = row
json.dump(out, open('/dev/stdout', 'w'), indent=1)
