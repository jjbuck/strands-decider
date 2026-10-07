"""Row routing (B8 generalized): which state rows to run at int8 so the int4 rounding error the decision sees falls most.
Per DEV request (train split, disjoint tasks):
 1. dense differentiable forward, capturing question->state attention at layers 3 and 7, and every GEMM input's int4 scale per row;
 2. backward of the margin -> per-row expected first-order int4 energy  e_t = sum_GEMMs ||dm/dX_t||^2 * s_t^2 / 12  (activation rounding,
    rotated per-token int4: error variance s^2/12 per value; the oracle score, needs a backward pass);
 3. for each rule and fraction f, choose f of the state rows (plus 4 sink rows), run the base format with those rows W8A8, record dm.
Rules: oracle (e_t), qattn3 / qattn7 (attention mass from question rows; computed from a dense pass here, i.e. an upper bound for a
cheap pre-pass), last (most recent rows), random.
python q1router.py --base w4q8 --n 150 -> ~/work/q1/router_{base}.json"""
import os, sys, json, time, argparse, math, zlib
sys.path[:0] = [os.path.expanduser('~/work/q1')]
import torch
import q1lib as QL, q1fmt as QF

ap = argparse.ArgumentParser(); ap.add_argument('--base', default='w4q8'); ap.add_argument('--n', type=int, default=150)
ap.add_argument('--fracs', default='0.05,0.1,0.2'); ap.add_argument('--rules', default='oracle,qattn3,qattn7,last,random')
a = ap.parse_args()
CF = json.load(open(os.path.expanduser('~/work/q1/cfgs.json')))
g = QL.Q1(); dev = g.dev
fm = QF.Fmt(g, CF[a.base])
_, B = QL.req_sets()
items = QL.dev_set(a.n, skip_rids=[b['rid'] for b in B])
fracs = [float(x) for x in a.fracs.split(',')]; rules = a.rules.split(',')
outf = os.path.expanduser(f'~/work/q1/router_{a.base}.json')
res = dict(base=a.base, rows=[])
t0 = time.time()
for di, it in enumerate(items):
    pr = g.prep(it['state'], it['q']); ids = pr['s'] + pr['q']; q0 = pr['q0']; T = len(ids)
    if q0 < 20: continue
    SC = {}
    st = {}

    def fwd_fn(i, k, x, xn, y):
        src = (xn if k in ('Win', 'Wgu') else x).detach().float()
        xr = g.rot_for(i, k)(src); am = xr.abs().amax(-1).clamp_min(1e-8)
        st[(i, k)] = ((am / 7.0) * 0.9).pow(2) / 12.0               # per-row int4 error variance per value

    E = torch.zeros(T, device=dev)

    def hook(i, k, gr):
        gi = gr.float() @ g.L[i][k].float(); gn = g.gain(i, k)
        if gn is not None: gi = gi * gn[None, :]
        E.add_(gi.pow(2).sum(1) * st.pop((i, k)))
    g.qfn = {}; g.fwd_fn = fwd_fn; g.hook_fn = hook; g.track = None
    g.cap_att = dict(layers=(3, 7), scores=SC)
    h = g.fwdg(ids, gfrom=0, q0=q0); lg = g.logits_g(h, pr)
    g.cap_att = None
    o = torch.argsort(lg.detach(), descending=True); t1, t2 = int(o[0]), int(o[1]); m0 = float(lg[t1] - lg[t2])
    (lg[t1] - lg[t2]).backward()
    g.fwd_fn = None; g.hook_fn = None
    del h, lg
    gen = torch.Generator(device='cpu'); gen.manual_seed(zlib.crc32(it['rid'].encode()))
    row = dict(rid=it['rid'], T=T, q0=q0, m0=m0, dm={})
    Es = E[:q0].clone()
    fm.install(g); fm.seed = zlib.crc32(it['rid'].encode())
    with torch.no_grad():
        g._row8 = None
        hq, _ = g.fwd(ids, q0=q0); lq = g.logits(hq, pr)
        row['dm']['none'] = float(lq[t1] - lq[t2]) - m0
        for rule in rules:
            for f in fracs:
                n = max(1, int(round(f * q0)))
                if rule == 'oracle': sc = Es
                elif rule.startswith('qattn'): sc = SC[int(rule[5:])].clone()
                elif rule == 'last': sc = torch.arange(q0, device=dev, dtype=torch.float32)
                else: sc = torch.rand(q0, generator=gen).to(dev)
                sc = sc.clone(); sc[:4] = float('inf')
                sel = torch.topk(sc, min(q0, n + 4)).indices
                m = torch.zeros(T, dtype=torch.bool, device=dev); m[sel] = True
                g._row8 = m
                hq, _ = g.fwd(ids, q0=q0); lq = g.logits(hq, pr)
                row['dm'][f'{rule}@{f}'] = float(lq[t1] - lq[t2]) - m0
                if rule == 'oracle': row.setdefault('oracle_share', {})[str(f)] = float(Es[sel].sum() / Es.sum())
        g._row8 = None
    g.qfn = {}
    res['rows'].append(row)
    if len(res['rows']) % 10 == 0:
        R = res['rows']
        summ = {k: math.sqrt(sum(r['dm'][k] ** 2 for r in R) / len(R)) for k in R[0]['dm']}
        print(len(R), f'{time.time()-t0:.0f}s', {k: round(v, 4) for k, v in summ.items()}, flush=True)
        json.dump(res, open(outf, 'w'))
R = res['rows']
res['rms_dm'] = {k: math.sqrt(sum(r['dm'][k] ** 2 for r in R) / len(R)) for k in R[0]['dm']}
json.dump(res, open(outf, 'w'))
print('done', {k: round(v, 4) for k, v in res['rms_dm'].items()}, flush=True)
