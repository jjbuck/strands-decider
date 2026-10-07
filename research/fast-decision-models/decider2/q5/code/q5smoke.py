"""Smoke test of every S5 path on layer 22 (needs cal/{bank,ret} for layer 22 from a tiny q5cal run, and h1 hess). Writes nothing persistent but cache/."""
import os, sys, json, time
sys.path[:0] = [os.path.expanduser('~/work/q5'), os.path.expanduser('~/work/q1')]
import torch
import q1lib as QL, q5lib as Q5
Q5.CACHE = os.path.expanduser('~/work/q5/cache_smoke')
g = QL.Q1(grad=False)
items = Q5.dev_items(4, 2)
ref = Q5.run_items(g, items)
B8 = json.load(open(os.path.expanduser('~/work/q1/cfgs.json')))['b8']
tests = []
for m in ('mag0', 'wanda', 'sgpt', 'sgptd', 'fisher'):
    tests.append((f'{m}.bf16n', dict(s24=dict(layers=[22], method=m, prec='bf16n'))))
for m in ('sgptd', 'fisher'):
    tests.append((f'{m}.bf16r', dict(s24=dict(layers=[22], method=m, prec='bf16r'))))
    tests.append((f'{m}.bf16r.s', dict(s24=dict(layers=[22], method=m, prec='bf16r', role='s'))))
for p in ('int8', 'int4'):
    tests.append((f'b8+sgptd.{p}', dict(fmt=B8, s24=dict(layers=[22], method='sgptd', prec=p))))
    tests.append((f'b8+mag0.{p}.s', dict(fmt=B8, s24=dict(layers=[22], method='mag0', prec=p, role='s'))))
tests.append(('nr.dsal', dict(neur=dict(layers=[22], method='dsal', frac=0.25))))
tests.append(('nr.dsalc', dict(neur=dict(layers=[22], method='dsal', frac=0.25, comp=True, dec_comp=True))))
tests.append(('nr.obsd', dict(neur=dict(layers=[22], method='obsd', frac=0.25))))
tests.append(('nr.var.global', dict(neur=dict(layers=[20, 21, 22], method='dsal', frac=0.25, **{'global': True}))))
tests.append(('b8+nr.obs', dict(fmt=B8, neur=dict(layers=[22], method='obs', frac=0.25))))
tests.append(('b8', dict(fmt=B8)))
for nm, cfg in tests:
    t0 = time.time()
    S = Q5.S5(g, cfg, tag=nm); S.install()
    lg = Q5.run_items(g, items); g.qfn = {}
    m = Q5.dev_metrics(ref, lg, items)['all']
    if 's24' in cfg:
        e = next(iter(S.sp.values()))
        z = (e['W'] if 'W' in e else e['Wr'] if 'Wr' in e else e['q']).float()
        zfrac = float((z == 0).float().mean())
    else: zfrac = None
    print(f'{nm:20s} KL {m["kl"]:.2e} TV {m["tv"]:.4f} flips {m["flips"]} zero-frac {zfrac} work {S.work()} {time.time()-t0:.1f}s', flush=True)
print('smoke ok')
