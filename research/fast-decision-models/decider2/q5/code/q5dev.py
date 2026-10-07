"""Q5 dev screening: structure configs on train-split DEV requests (tau tasks disjoint from calibration; banking + retail).
Metrics vs the in-runtime bf16 dense logits (and vs the config's base format alone, when it has one): KL, TV, argmax flips, per domain.
python q5dev.py --cfgs cfgs.json --tags a,b [--out ~/work/q5/dev_res.json] [--nb 160 --nr 96]
"""
import os, sys, json, time, argparse
sys.path[:0] = [os.path.expanduser('~/work/q5'), os.path.expanduser('~/work/q1')]
import torch
import q1lib as QL, q5lib as Q5

ap = argparse.ArgumentParser(); ap.add_argument('--cfgs', required=True); ap.add_argument('--tags', required=True)
ap.add_argument('--out', default=os.path.expanduser('~/work/q5/dev_res.json')); ap.add_argument('--nb', type=int, default=160); ap.add_argument('--nr', type=int, default=96)
ap.add_argument('--redo', action='store_true'); ap.add_argument('--prefix', default='', help='layers whose input residual is cached per base format, e.g. 4,8,12,16,20')
a = ap.parse_args()
CF = json.load(open(a.cfgs))
g = QL.Q1(grad=False)
items = Q5.dev_items(a.nb, a.nr)
print('dev items', len(items), flush=True)
os.makedirs(Q5.W5, exist_ok=True)
res = json.load(open(a.out)) if os.path.exists(a.out) else {}


def ref_for(name, fmt):
    fn = f'{Q5.W5}/devref_{name}_{a.nb}_{a.nr}.pt'
    if os.path.exists(fn): return torch.load(fn)
    S = Q5.S5(g, dict(fmt=fmt), tag=name) if fmt else None
    if S: S.install()
    else: g.qfn = {}
    t0 = time.time(); lg = Q5.run_items(g, items); g.qfn = {}
    print(f'ref {name} {time.time()-t0:.0f}s', flush=True)
    torch.save(lg, fn); return lg


dense = ref_for('dense', None)
PFX = {}                     # base name -> per-item prefix residuals (cpu bf16), built on first use
PL = [int(x) for x in a.prefix.split(',')] if a.prefix else []


def prefix_for(cfg):
    if not PL: return None, 0
    L0 = Q5.first_layer(cfg); st = max([L for L in PL if L <= L0], default=0)
    if st == 0: return None, 0
    name = cfg.get('fmt_name') or ('dense' if not cfg.get('fmt') else None)
    if name is None: return None, 0
    if cfg.get('fcal'): name += '.f' + cfg['fcal']
    if name not in PFX:
        t0 = time.time()
        if cfg.get('fmt'): Sb = Q5.S5(g, dict(fmt=cfg['fmt'], fcal=cfg.get('fcal')), tag=name); Sb.install()
        else: g.qfn = {}
        PFX[name] = Q5.build_prefix(g, items, PL); g.qfn = {}
        print(f'prefix {name} layers {PL} built {time.time()-t0:.0f}s', flush=True)
    return PFX[name], st


for tag in a.tags.split(','):
    if tag in res and not a.redo: print('skip', tag); continue
    cfg = CF[tag]; t0 = time.time()
    pfx, st = prefix_for(cfg)
    S = Q5.S5(g, cfg, tag=tag); S.install()
    tb = time.time() - t0
    lg = Q5.run_items(g, items, pfx, st); g.qfn = {}
    r = dict(cfg=cfg, vs_dense=Q5.dev_metrics(dense, lg, items), work=S.work(), stats=S.stats, build_s=tb, run_s=time.time() - t0 - tb)
    if cfg.get('fmt'):
        import hashlib; fname = cfg.get('fmt_name') or ('fmt_' + hashlib.md5(json.dumps(cfg['fmt'], sort_keys=True).encode()).hexdigest()[:10])
        base = ref_for(fname, cfg['fmt'])
        r['vs_fmt'] = Q5.dev_metrics(base, lg, items)
    res[tag] = r
    v = r['vs_dense']
    print(f"### {tag}: bank KL {v['bank']['kl']:.2e} flips {v['bank']['flips']}/{v['bank']['n']} | ret KL {v['ret']['kl']:.2e} flips {v['ret']['flips']}/{v['ret']['n']} | "
          f"all TV {v['all']['tv']:.4f} | work {r['work']} | {r['build_s']:.0f}+{r['run_s']:.0f}s", flush=True)
    if 'vs_fmt' in r:
        w = r['vs_fmt']; print(f"    vs fmt: bank KL {w['bank']['kl']:.2e} flips {w['bank']['flips']} | ret KL {w['ret']['kl']:.2e} flips {w['ret']['flips']}", flush=True)
    json.dump(res, open(a.out + '.tmp', 'w'), indent=1); os.replace(a.out + '.tmp', a.out)
    del S; torch.cuda.empty_cache()
print('done', flush=True)
