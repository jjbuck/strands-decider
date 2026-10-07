"""M1 untrained conversion measurements (train-split data only; evalkit is scored separately by m1eval.py).
  python m1conv.py check            : teacher path vs evalkit hobson refs on 20 items; student with no layer converted == teacher
  python m1conv.py local  OUT.json  : teacher-forced per-layer fidelity of the converted mixer (relMSE / cosine of the residual contribution
                                      vs the GDN's) for every init variant x temperature tau; picks tau per layer and variant
  python m1conv.py e2e    OUT.json  : all 18 converted at once, per variant (best tau): dev agreement / agree_sd / TV / KL / hidden relMSE;
                                      one layer at a time; bottom-up cumulative
"""
import os, sys, json, time, random, collections
sys.path[:0] = [os.path.expanduser('~/work/m1')]
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
import m1lib as ML

mode = sys.argv[1]; OUT = os.path.expanduser(sys.argv[2]) if len(sys.argv) > 2 else None
WM = os.path.expanduser('~/work/m1')
m = ML.M1()
VARIANTS = {
    'plain': dict(conv=True, rope=True, beta=False, decay=False),
    'beta': dict(conv=True, rope=True, beta=True, decay=False),
    'decay': dict(conv=True, rope=True, beta=False, decay=True),
    'full': dict(conv=True, rope=True, beta=True, decay=True),
    'full136': dict(conv=True, rope=True, beta=True, decay=True, slots=0),
    'full_norope': dict(conv=True, rope=False, beta=True, decay=True),
    'full_noconv': dict(conv=False, rope=True, beta=True, decay=True),
    'plain_norope': dict(conv=True, rope=False, beta=False, decay=False),
}
TAUS = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
pool, devset, v5, _ = ML.load_data(maxtok=5000, ndev=100, seed=0)
res = json.load(open(OUT)) if OUT and os.path.exists(OUT) else {}


def save():
    if OUT: json.dump(res, open(OUT, 'w'), indent=1)


if mode == 'check':
    import evalkit as EK
    refs = {}
    for s in ('REAL-agree', 'JB-all'):
        for iid, r in EK.load_refs(s).items(): refs[iid] = r.get('hobson', {})
    items = collections.OrderedDict()
    for suite, iid, q, st, spec in EK.all_question_items(['REAL-agree', 'JB-hard']):
        items.setdefault(iid, (st, collections.OrderedDict()))[1][q] = spec
    its = list(items.items())[:12] + [x for x in items.items() if x[0] not in dict(list(items.items())[:12])][-8:]
    ag = n = 0; maxd = 0.0; same = 0
    with torch.no_grad():
        for iid, (st, qs) in its:
            names = list(qs); rq = m.prep(st, [qs[k] for k in names])
            (lt, _), _ = m.run(rq, student=False, keep=())
            (ls, _), _ = m.run(rq, student=True, keep=())          # nothing converted, no LoRA: must equal the teacher
            for qn, a, b, qx in zip(names, lt, ls, rq['qs']):
                maxd = max(maxd, float((a.exp() - b.exp()).abs().max())); same += int(a.argmax() == b.argmax()); n += 1
                rd = refs.get(iid, {}).get(qn)
                if rd: ag += int(qx['labels'][int(a.argmax())] == max(rd, key=rd.get))
    r = dict(n=n, teacher_vs_refs_argmax=ag, student0_vs_teacher_argmax=same, student0_maxdp=maxd)
    print('check', r, flush=True); res['check'] = r; save(); sys.exit(0)

if mode == 'local':
    m.convert(ML.GDN, tau=1.0)
    rng = random.Random(5)
    reqs = [r for r in pool[:400] if 300 <= r['n'] <= 3000]
    rng.shuffle(reqs); reqs = reqs[:16]
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    t0 = time.time()
    for k, r in enumerate(reqs):
        names = sorted(r['questions'])[:2]
        rq = m.prep(r['state'], [r['questions'][n] for n in names]); lay = m.layout(rq)
        out = m.local(lay, ML.GDN, cfgs=list(VARIANTS.items()), taus={nm: TAUS for nm in VARIANTS})
        for key, d in out.items():
            for i, (e, cs) in d.items(): acc[key][i].append((e, cs))
        print(k, lay['T'], f'{time.time() - t0:.0f}s', flush=True)
    tab = {}
    for key, d in acc.items():
        tab[key] = {str(i): [sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v)] for i, v in d.items()}
    best = {}
    for nm in VARIANTS:
        bt = {}
        for i in ML.GDN:
            cands = [(tab[f'{nm}@{t}'][str(i)][0], t) for t in TAUS]
            e, t = min(cands); bt[str(i)] = dict(tau=t, relmse=e, cos=tab[f'{nm}@{t}'][str(i)][1])
        best[nm] = dict(layers=bt, mean_relmse=sum(v['relmse'] for v in bt.values()) / len(bt))
        print(nm, 'mean relMSE at best tau per layer', round(best[nm]['mean_relmse'], 4), {i: (v['tau'], round(v['relmse'], 3)) for i, v in bt.items()}, flush=True)
    res['local'] = dict(n_req=len(reqs), taus=TAUS, table=tab, best=best); save(); sys.exit(0)

if mode == 'e2e':
    loc = json.load(open(os.path.expanduser(sys.argv[3])))['local']['best']
    ref = ML.make_devref(m, devset, f'{WM}/devref.pt')
    names = sys.argv[4].split(',') if len(sys.argv) > 4 else ['plain', 'full', 'decay', 'beta', 'full_norope', 'full_noconv']
    m.convert(ML.GDN, tau=1.0)

    def set_variant(nm):
        m.cfg = dict(VARIANTS[nm])
        with torch.no_grad():
            for i in ML.GDN: m.C[str(i)].qg.fill_(loc[nm]['layers'][str(i)]['tau'])

    res.setdefault('e2e', {})
    for nm in names:
        if nm in res['e2e']: continue
        set_variant(nm); m.active = set(ML.GDN); t0 = time.time()
        r = ML.dev_eval(m, devset, ref); r['t'] = round(time.time() - t0)
        res['e2e'][nm] = r; print('all18', nm, r, flush=True); save()
    best_nm = sys.argv[5] if len(sys.argv) > 5 else min(names, key=lambda k: res['e2e'][k]['tv'])
    set_variant(best_nm)
    res.setdefault('single', {}); res['single_variant'] = best_nm
    for i in ML.GDN:
        if str(i) in res['single']: continue
        m.active = {i}; r = ML.dev_eval(m, devset, ref, limit=40); res['single'][str(i)] = r
        print('single', i, {k: r[k] for k in ('agree', 'agree_sd', 'tv')}, flush=True); save()
    res.setdefault('cum', {})
    for kk in (3, 6, 9, 12, 15):
        if str(kk) in res['cum']: continue
        m.active = set(ML.GDN[:kk]); r = ML.dev_eval(m, devset, ref, limit=40); res['cum'][str(kk)] = r
        print('bottom-up', kk, {k: r[k] for k in ('agree', 'agree_sd', 'tv')}, flush=True); save()
    save(); sys.exit(0)
