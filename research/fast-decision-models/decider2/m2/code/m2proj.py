"""latency tables + projections (laptop, pure python). Projection [A] (J9's method, calibrated to FAST_DECISION_MODEL sec 9):
t_X = (t_A10G - t_mem) * r_c + t_mem * r_m, t_mem = 6.3 ms (2B bf16 weights streamed once per pass at 600 GB/s);
r_c = .503 / .214 / .170 (3090 / 4090 / 5090, fp16 accumulation), r_m = 600 / bandwidth = .641 / .595 / .335."""
import json, os, sys, statistics as st
RC = {'3090': 0.503, '4090': 0.214, '5090': 0.170}; RM = {'3090': 0.641, '4090': 0.595, '5090': 0.335}; TM = 6.3


def proj(t):
    return {k: round((max(t - TM, 0)) * RC[k] + min(t, TM) * RM[k], 1) for k in RC}


R = os.path.expanduser('~/decider2/m2/res/')
GF = sys.argv[1] if len(sys.argv) > 1 else 'lat_grid.json'
g = json.load(open(R + GF)) if os.path.exists(R + GF) else {}
ks = g.get('meta', {}).get('ks', [8, 12])
cols = ['hob1', 'hobB'] + [f'dtG{k}' for k in ks] + [f'm2k{k}_c{c}' for k in ks + [24] for c in (0.0, 0.55)]
for q in (1, 4):
    print(f'\n### {q} question(s): A10G median / p95 ms')
    print('| T | ' + ' | '.join(cols) + ' |'); print('|---' * (len(cols) + 1) + '|')
    for T in (64, 256, 1000, 4000):
        cells = []
        for c in cols:
            r = g.get(f'Q{q}_T{T}_{c}')
            cells.append(f"{r['median']:.1f} / {r['p95']:.1f}" if r else '-')
        print(f'| {T} | ' + ' | '.join(cells) + ' |')
print('\n### projections, 1 question (A10G measured -> 3090 / 4090 / 5090 arithmetic)')
for T in (64, 256, 1000, 4000):
    out = []
    for c in ['hob1'] + [f'm2k{k}_c{c}' for k in ks for c in (0.0, 0.55)]:
        r = g.get(f'Q1_T{T}_{c}')
        if r: p = proj(r['median']); out.append(f"{c} {r['median']:.1f} -> {p['3090']} / {p['4090']} / {p['5090']}")
    print(f'T={T}: ' + ' | '.join(out))
if os.path.exists(R + 'lat_real.json'):
    rr = json.load(open(R + 'lat_real.json'))['rows']
    keys = ['hob1'] + [k for k in rr[0] if k.startswith('m2ck') and not k.endswith('p95')]
    for nm, sel in (('all 84', rr), ('banking', [x for x in rr if x['domain'] == 'banking_knowledge']), ('retail', [x for x in rr if x['domain'] == 'retail'])):
        if not sel: continue
        s = f"real {nm:<8} n={len(sel):<3} state tok {st.mean([x['state'] for x in sel]):.0f} live {st.mean([x['live'] for x in sel]):.0f} compiled {st.mean([x['compiled'] for x in sel]):.0f} |"
        for k in keys:
            v = [x[k] for x in sel]
            s += f" {k} mean {st.mean(v):.1f} med {st.median(v):.1f} |"
        for k in keys[1:]:
            s += f" speedup of means {k}: {st.mean([x['hob1'] for x in sel]) / st.mean([x[k] for x in sel]):.2f}, median per-req {st.median([x['hob1'] / x[k] for x in sel]):.2f} |"
        print(s)
