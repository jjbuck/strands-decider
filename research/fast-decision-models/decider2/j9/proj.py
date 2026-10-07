"""latency tables + projections (laptop). Projection [A]: t_X = (t_A10G - t_mem) * r_c + t_mem * r_m, t_mem = 6.3 ms (2B bf16 weights
streamed once per pass at 600 GB/s); r_c fitted so hobson bf16 at 1000 tokens reproduces FAST_DECISION_MODEL.md section 9
(3090 27.4, 4090 13.7, 5090 10.0 ms from 52.7): r_c = .503 / .214 / .170 (fp16 accumulation); r_m = 600/BW = .641 / .595 / .335."""
import json, sys, os, statistics as st
RC = {'3090': 0.503, '4090': 0.214, '5090': 0.170}; RM = {'3090': 0.641, '4090': 0.595, '5090': 0.335}; TM = 6.3


def proj(t):
    return {k: round((max(t - TM, 0)) * RC[k] + min(t, TM) * RM[k], 1) for k in RC}


g = json.load(open(os.path.expanduser('~/decider2/j9/lat_grid.json')))
print('| T | q | hobson native (median / p95) | compiled c=.37 | compiled c=.55 | speedup .37 / .55 |')
print('|---|---|---|---|---|---|')
rows = []
for q in (1, 4):
    for T in (64, 128, 256, 400, 1000, 4000):
        n = g.get(f'native_q{q}_T{T}'); a = g.get(f'R_c0.37_q{q}_T{T}'); b = g.get(f'R_c0.55_q{q}_T{T}')
        if not n: continue
        f = lambda r: f"{r['median']:.1f} / {r['p95']:.1f}" if r else '-'
        sp = lambda r: f"{n['median'] / r['median']:.2f}" if r else '-'
        print(f"| {T} | {q} | {f(n)} | {f(a)} | {f(b)} | {sp(a)} / {sp(b)} |")
        rows.append(dict(T=T, q=q, native=n['median'], c37=a['median'] if a else None, c55=b['median'] if b else None))
print()
print('projections (1 question):')
print('| T | hobson A10G / 3090 / 4090 / 5090 | compiled c=.55 A10G / 3090 / 4090 / 5090 |')
for r in rows:
    if r['q'] != 1: continue
    pn = proj(r['native']); pc = proj(r['c55']) if r['c55'] else None
    print(f"| {r['T']} | {r['native']:.1f} / {pn['3090']} / {pn['4090']} / {pn['5090']} | " + (f"{r['c55']:.1f} / {pc['3090']} / {pc['4090']} / {pc['5090']} |" if pc else '- |'))
if os.path.exists(os.path.expanduser('~/decider2/j9/lat_real.json')):
    r = json.load(open(os.path.expanduser('~/decider2/j9/lat_real.json')))
    rr = r['rows']
    for nm, sel in (('all', rr), ('REAL-agree', [x for x in rr if x['suite'] == 'REAL-agree']), ('LONG', [x for x in rr if x['suite'] == 'LONG']),
                    ('banking', [x for x in rr if x['domain'] == 'banking_knowledge']), ('retail', [x for x in rr if x['domain'] == 'retail'])):
        if not sel: continue
        nat = [x['native'] for x in sel]; com = [x['compiled'] for x in sel]
        print(f"real {nm:<10} n={len(sel):<3} tokens mean {st.mean([x['T'] for x in sel]):.0f} live {st.mean([x['live'] for x in sel]):.0f} | native median {st.median(nat):.1f} mean {st.mean(nat):.1f} p95 {sorted(nat)[int(.95*(len(nat)-1))]:.1f} | compiled median {st.median(com):.1f} mean {st.mean(com):.1f} p95 {sorted(com)[int(.95*(len(com)-1))]:.1f} | mean speedup {st.mean(nat)/st.mean(com):.2f} median per-req {st.median([a/b for a, b in zip(nat, com)]):.2f}")
