"""Generate ~/decider2/q5/code/cfgs_q5.json: named structure configs for q5dev.py / q5kit.py (laptop, pure python)."""
import json, os
B8 = {"a_s": 8, "a_q": 8, "map": {"23.Wd": "bf16", "0.Wo": "bf16", "7.Wo": "bf16", "11.Wo": "bf16", "10.Wo": "bf16", "23.Wo": "bf16", "12.Wo": "bf16", "9.Wo": "bf16"}}
W4Q8 = {"a_s": 4, "a_q": 8}
ALL = ["Win", "Wo", "Wgu", "Wd"]
C = {"dense": None, "b8": {"fmt": B8, "fmt_name": "b8"}, "w4q8": {"fmt": W4Q8, "fmt_name": "w4q8"}}


def rng(a, b): return list(range(a, b + 1))


def s24(name, layers, method, prec='bf16n', role='all', cal='gen', gemms=ALL, fmt=None, fmt_name=None, **kw):
    c = {"s24": dict(layers=layers, gemms=gemms, role=role, prec=prec, method=method, cal=cal, **kw)}
    if fmt: c["fmt"] = fmt; c["fmt_name"] = fmt_name
    C[name] = c


LR = {'L12-22': rng(12, 22), 'L13-23': rng(13, 23), 'L16-22': rng(16, 22), 'L8-22': rng(8, 22), 'L8-23': rng(8, 23), 'L4-23': rng(4, 23), 'L0-23': rng(0, 23),
      'L12-23': rng(12, 23), 'L1-23': rng(1, 23), 'L4-22': rng(4, 22), 'L16-23': rng(16, 23), 'L14-23': rng(14, 23)}
# B11 set 1: bf16, layers 12-22, all rows: selection method and basis
for m in ('mag0', 'wanda', 'sgpt', 'sgptd', 'fisher'):
    for p in ('bf16n', 'bf16r'):
        s24(f's24.L12-22.{m}.{p}', LR['L12-22'], m, p)
for m in ('sgpt', 'sgptd'):
    s24(f's24.L12-22.{m}.bf16n.s', LR['L12-22'], m, 'bf16n', role='s')
    s24(f's24.L12-22.{m}.bf16r.s', LR['L12-22'], m, 'bf16r', role='s')
s24('s24.L12-22.sgptd.bf16n.mlp', LR['L12-22'], 'sgptd', gemms=['Wgu', 'Wd'])
s24('s24.L12-22.sgptd.bf16n.mix', LR['L12-22'], 'sgptd', gemms=['Win', 'Wo'])
# B11 set 2: with formats (b8 base); int8 2:4 and int4 2:4
for m in ('mag0', 'sgpt', 'sgptd', 'fisher'):
    for role in ('all', 's'):
        for p in ('int8', 'int4'):
            s24(f'b8+s24.L12-22.{m}.{p}.{role}', LR['L12-22'], m, p, role=role, fmt=B8, fmt_name='b8')
# widening (filled for every method so the queue can pick)
for L in ('L13-23', 'L16-22', 'L8-22', 'L8-23', 'L4-23', 'L0-23', 'L12-23', 'L1-23', 'L4-22'):
    for m in ('sgpt', 'sgptd', 'fisher'):
        for role in ('all', 's'):
            s24(f's24.{L}.{m}.bf16n.{role}', LR[L], m, 'bf16n', role=role)
            for p in ('int8', 'int4'):
                s24(f'b8+s24.{L}.{m}.{p}.{role}', LR[L], m, p, role=role, fmt=B8, fmt_name='b8')
# int8 2:4 in the natural basis (+ smoothing), b8 base
for m in ('sgpt', 'sgptd', 'fisher', 'mag0'):
    for role in ('all', 's'):
        for sm in (0.0, 0.5):
            s24(f'b8+s24.L12-22.{m}.int8n{sm:g}.{role}', LR['L12-22'], m, 'int8n', role=role, fmt=B8, fmt_name='b8', smooth=sm)
# int4 with Ampere's pair-wise mask (keep 2 of 4 adjacent nibble pairs)
for m in ('sgptd', 'fisher', 'sgpt', 'mag0'):
    for role in ('all', 's'):
        for L in ('L12-22', 'L12-23', 'L13-23', 'L14-23', 'L16-23'):
            s24(f'b8+s24.{L}.{m}.int4p.{role}', LR[L] if L in LR else rng(int(L[1:3]), 23), m, 'int4p', role=role, fmt=B8, fmt_name='b8')
# per-layer scans (one layer at a time), for greedy selection of the sparse set
for i in range(24):
    for m in ('sgptd', 'fisher', 'sgpt'):
        s24(f's24.L{i}.{m}.bf16n', [i], m, 'bf16n')
        s24(f's24.L{i}.{m}.bf16n.s', [i], m, 'bf16n', role='s')
        s24(f'b8+s24.L{i}.{m}.int8', [i], m, 'int8', fmt=B8, fmt_name='b8')
        s24(f'b8+s24.L{i}.{m}.int8.s', [i], m, 'int8', role='s', fmt=B8, fmt_name='b8')
# B7: banking-calibrated versions of everything above
for k in list(C):
    v = C[k]
    if v and 's24' in v:
        v2 = json.loads(json.dumps(v)); v2['s24']['cal'] = 'bank'; C[k + '.bank'] = v2


# B12: neuron removal
def neur(name, layers, method, cal='gen', fmt=None, fmt_name=None, **kw):
    c = {"neur": dict(layers=layers, method=method, cal=cal, **kw)}
    if fmt: c["fmt"] = fmt; c["fmt_name"] = fmt_name
    C[name] = c


for cal in ('gen', 'bank'):
    sfx = '' if cal == 'gen' else '.bank'
    for tol in (1e-3, 1e-2, 3e-2, 1e-1):
        neur(f'nr.L0-23.dead{tol:g}{sfx}', LR['L0-23'], 'dead', cal=cal, tol=tol)
    for L in ('L12-23', 'L0-23', 'L4-23', 'L8-23', 'L16-23'):
        for f in (0.05, 0.1, 0.25, 0.5):
            for m in ('var', 'dsal'):
                neur(f'nr.{L}.{m}.f{f:g}{sfx}', LR[L], m, cal=cal, frac=f)
            neur(f'nr.{L}.dsalc.f{f:g}{sfx}', LR[L], 'dsal', cal=cal, frac=f, comp=True, dec_comp=True)
            neur(f'nr.{L}.obs.f{f:g}{sfx}', LR[L], 'obs', cal=cal, frac=f)
            neur(f'nr.{L}.obsd.f{f:g}{sfx}', LR[L], 'obsd', cal=cal, frac=f)
    for f in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4):
        neur(f'nr.G0-23.dsal.f{f:g}{sfx}', LR['L0-23'], 'dsal', cal=cal, frac=f, **{'global': True})
        neur(f'nr.G12-23.dsal.f{f:g}{sfx}', LR['L12-23'], 'dsal', cal=cal, frac=f, **{'global': True})
# B7 precision: per-deployment GPTQ calibration of the formats themselves
for fc in ('bank', 'ret'):
    C[f'b8.f{fc}'] = {"fmt": B8, "fmt_name": f"b8.f{fc}", "fcal": fc}
    C[f'w4q8.f{fc}'] = {"fmt": W4Q8, "fmt_name": f"w4q8.f{fc}", "fcal": fc}
json.dump(C, open(os.path.expanduser('~/decider2/q5/code/cfgs_q5.json'), 'w'), indent=0)
print(len(C), 'configs')
