"""Q1 first-order pass for one format (B1.3 predictor, B2 coherence, B6 certificate, B8 token classes, B10 option rows).

Per request (train-split DEV, disjoint tau tasks from the spectrum set A):
 1. dense differentiable forward; at every GEMM the LOCAL error of the format on the dense trajectory, e = Y_fmt(x) - Y_bf16(x);
 2. backward of the decision margin m = l_top1 - l_top2 (dense top-2): a_t = <dm/dY_t, e_t> per row and GEMM;
    first-order prediction dm_pred = sum_GEMMs sum_t a_t; coherence per GEMM and role: S = sum_t a_t, Q = sum_t a_t^2;
    B8: sum of a_t^2 per token class; B10: option-row input-side gradients split into common (mean over options) and difference parts;
 3. quantized forward (the format everywhere): actual dm, flip; B6 certificate: c = sum_GEMMs sum_t e_t^T Gbar e_t with e_t the error the
    runtime knows (quantized output minus the dense GEMM of the same input) and Gbar = top-r eigenpairs of G_out(all) / mean rows per request.
python q1fo.py --spec '{"a_s":4,"a_q":4}' --tag w4a4 --n 400
"""
import os, sys, json, time, argparse, math, zlib
sys.path[:0] = [os.path.expanduser('~/work/q1')]
import torch
import q1lib as QL, q1fmt as QF, q1tok as QT

ap = argparse.ArgumentParser()
ap.add_argument('--spec', default='{}'); ap.add_argument('--cfg', default=''); ap.add_argument('--tag', required=True); ap.add_argument('--n', type=int, default=400)
ap.add_argument('--r', type=int, default=64); ap.add_argument('--out', default=os.path.expanduser('~/work/q1/fo'))
ap.add_argument('--b10', type=int, default=1); ap.add_argument('--b10from', type=int, default=16)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
outf = f'{a.out}/{a.tag}.json'
g = QL.Q1(); dev = g.dev
spec = json.load(open(os.path.expanduser('~/work/q1/cfgs.json')))[a.cfg] if a.cfg else json.loads(a.spec)
fm = QF.Fmt(g, spec)
_, B = QL.req_sets()
items = QL.dev_set(a.n, skip_rids=[b['rid'] for b in B])
print('dev items', len(items), flush=True)
tok = g.p.tok
# certificate eigenpairs: G_out(all) top-r, normalized by the mean number of rows per request in set A
meta = []
for f in sorted(os.listdir(QF.SPEC_DIR)):
    if f.startswith('grp') and f.endswith('.json'): meta = json.load(open(f'{QF.SPEC_DIR}/{f}'))['meta']; break
rows_mean = sum(m['T'] for m in meta) / max(1, len(meta)) if meta else 1500.0
CV = {}
for i in range(24):
    for k in QL.GEMMS:
        sv = torch.load(f'{QF.SPEC_DIR}/L{i}_{k}.pt', map_location=dev)
        d = sv[('out', 'all')]
        CV[(i, k)] = (d['V'][:, :a.r].float().contiguous(), (d['ev'][:a.r].float().clamp_min(0) / rows_mean).to(dev))
res = json.load(open(outf)) if os.path.exists(outf) else dict(spec=spec, tag=a.tag, reqs=[], work=fm.work())
done = {r['rid'] for r in res['reqs']}
coh = res.setdefault('coh', {})                       # 'i.k' -> role -> [sum S^2, sum Q, n]
cls_acc = res.setdefault('cls', {})                   # class -> [rows, sum a^2, sum |a|]
b10 = {}                                              # (i,k) -> [Gcomm, Gdiff]
t0 = time.time()
for di, it in enumerate(items):
    if it['rid'] in done: continue
    pr = g.prep(it['state'], it['q']); ids = pr['s'] + pr['q']; q0 = pr['q0']; T = len(ids)
    opt_abs = [q0 + o for o in pr['opt']]
    # ---- 1. dense forward with local errors
    E = {}
    fm.seed = zlib.crc32(it['rid'].encode())

    def fwd_fn(i, k, x, xn, y):
        with torch.no_grad():
            yq = fm(g, i, k, x.detach(), None if xn is None else xn.detach())
            E[(i, k)] = (yq.float() - y.detach().float()).to(torch.bfloat16)
    g.qfn = {}; g.fwd_fn = fwd_fn; g.hook_fn = None; g.track = None
    h = g.fwdg(ids, gfrom=0, q0=q0); lg = g.logits_g(h, pr)
    p = torch.softmax(lg.detach(), -1); order = torch.argsort(lg.detach(), descending=True)
    t1, t2 = int(order[0]), int(order[1])
    m0 = float(lg[t1] - lg[t2]); wF = float(p[t1] * p[t2])
    # ---- 2. backward of the margin
    A_ = {}

    def hook(i, k, gr):
        e = E.pop((i, k))
        A_[(i, k)] = (gr.float() * e.float()).sum(1)
        if a.b10 and i >= a.b10from:
            go = gr.float()[torch.tensor(opt_abs, device=dev)]
            gi = go @ g.L[i][k].float(); gn = g.gain(i, k)
            if gn is not None: gi = gi * gn[None, :]
            mu = gi.mean(0, keepdim=True); df = gi - mu
            acc = b10.get((i, k))
            if acc is None: acc = b10[(i, k)] = [torch.zeros(gi.shape[1], gi.shape[1], device=dev), torch.zeros(gi.shape[1], gi.shape[1], device=dev)]
            acc[0].addmm_(mu.t(), mu, alpha=wF * gi.shape[0]); acc[1].addmm_(df.t(), df, alpha=wF)
    g.fwd_fn = None; g.hook_fn = hook
    (lg[t1] - lg[t2]).backward()
    g.hook_fn = None; E.clear()
    pred = sum(float(v.sum()) for v in A_.values())
    pred_s = sum(float(v[:q0].sum()) for v in A_.values()); pred_q = pred - pred_s
    per_layer = [sum(float(A_[(i, k)].sum()) for k in QL.GEMMS) for i in range(24)]
    for (i, k), v in A_.items():
        for role, sl in (('s', slice(0, q0)), ('q', slice(q0, T))):
            vv = v[sl]
            if vv.numel() == 0: continue
            S = float(vv.sum()); Q = float(vv.pow(2).sum())
            c = coh.setdefault(QL.kname(i, k), {}).setdefault(role, [0.0, 0.0, 0])
            c[0] += wF * S * S; c[1] += wF * Q; c[2] += 1
    rowa = torch.stack(list(A_.values())).sum(0)                  # per-row total first-order contribution
    rowa2 = torch.stack([v.pow(2) for v in A_.values()]).sum(0)    # per-row sum over GEMMs of a^2
    cl = QT.classify(tok, ids)
    for t in range(q0):
        c = cls_acc.setdefault(cl[t], [0, 0.0, 0.0]); c[0] += 1; c[1] += wF * float(rowa2[t]); c[2] += wF * abs(float(rowa[t]))
    c = cls_acc.setdefault('_question', [0, 0.0, 0.0]); c[0] += T - q0; c[1] += wF * float(rowa2[q0:].sum()); c[2] += wF * float(rowa[q0:].abs().sum())
    pos = res.setdefault('pos', {})                    # state-row first-order energy by position: sinks, last-L rows, rest
    for nm, sl in (('first4', slice(0, min(4, q0))), ('last64', slice(max(0, q0 - 64), q0)), ('last128', slice(max(0, q0 - 128), q0)),
                   ('last256', slice(max(0, q0 - 256), q0)), ('all_state', slice(0, q0)), ('question', slice(q0, T))):
        c = pos.setdefault(nm, [0, 0.0]); c[0] += sl.stop - sl.start; c[1] += wF * float(rowa2[sl].sum())
    if q0 > 0:                                         # oracle: share of state-row first-order energy in the top f of state rows
        srt = torch.sort(rowa2[:q0], descending=True).values; cs = torch.cumsum(srt, 0)
        for f in (0.01, 0.05, 0.1, 0.25):
            c = pos.setdefault(f'oracle_top{f}', [0, 0.0]); n_ = max(1, int(round(f * q0))); c[0] += n_; c[1] += wF * float(cs[n_ - 1])
    del A_, h, lg
    # ---- 3. quantized forward + certificate
    cert = [0.0] * 24; enorm = [0.0] * 24

    def cert_fn(i, k, x, xn, y):
        e = y.float() - (x @ g.L[i][k].t()).float()
        V, ev = CV[(i, k)]
        if V.shape[0] != e.shape[1]: return
        z = e @ V
        cert[i] += float((z.pow(2) * ev[None, :]).sum()); enorm[i] += float(e.pow(2).sum())
    fm.install(g); g.fwd_fn = cert_fn
    with torch.no_grad():
        hq = g.fwdg(ids, gfrom=99, q0=q0); lq = g.logits_g(hq, pr)
    g.qfn = {}; g.fwd_fn = None
    mq = float(lq[t1] - lq[t2]); flip = int(torch.argmax(lq)) != t1
    pq = torch.softmax(lq, -1)
    res['reqs'].append(dict(rid=it['rid'], task=it['task'], q=it['qname'], T=T, q0=q0, n=len(p), m0=m0, dm=mq - m0, pred=pred, pred_s=pred_s, pred_q=pred_q,
                            flip=flip, wF=wF, tv=0.5 * float((pq - p).abs().sum()), per_layer=per_layer, cert=cert, enorm=enorm))
    if len(res['reqs']) % 25 == 0 or di == len(items) - 1:
        json.dump(res, open(outf + '.tmp', 'w')); os.replace(outf + '.tmp', outf)
        rq = res['reqs']; import statistics as St
        xs = [r['pred'] for r in rq]; ys = [r['dm'] for r in rq]
        mx, my = St.mean(xs), St.mean(ys); cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs); vy = sum((y - my) ** 2 for y in ys)
        print(f'{a.tag} {len(rq)}/{len(items)} T={T} {time.time()-t0:.0f}s flips {sum(r["flip"] for r in rq)} corr(pred,dm) {cov/math.sqrt(vx*vy+1e-30):.3f} '
              f'slope {cov/(vx+1e-30):.3f} mem {torch.cuda.max_memory_allocated()/1e9:.1f}G', flush=True)
if a.b10 and b10:
    o = {}
    for (i, k), (Gc, Gd) in b10.items():
        o[QL.kname(i, k)] = {}
        for nm, M in (('common', Gc), ('diff', Gd)):
            ev = torch.linalg.eigvalsh(0.5 * (M + M.t())).flip(0).clamp_min(0).double().cpu(); t = float(ev.sum()); c = torch.cumsum(ev, 0)
            rk = lambda f: int(torch.searchsorted(c, torch.tensor(f * t, dtype=c.dtype)).item()) + 1 if t > 0 else 0
            o[QL.kname(i, k)][nm] = dict(trace=t, r90=rk(.9), r99=rk(.99))
    res['b10'] = o
json.dump(res, open(outf + '.tmp', 'w')); os.replace(outf + '.tmp', outf)
print('done', a.tag, f'{time.time()-t0:.0f}s', flush=True)
