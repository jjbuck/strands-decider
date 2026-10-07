"""J5 weight-only quantization of hobson-v19 for the fold runtime (h2 qrt, prec='bf16', d1 lean2 fold kernels).

The fold runtime's four GEMMs per layer read: Win_f = Win*(1+g_in) with input xn = x*rsqrt(mean x^2+eps) (gain-free norm),
Wo with input y (gated norm / attention gate output), Wgu_f (gate/up interleaved, *(1+g_post)) with input xn, Wd with input m = SwiGLU.
Weight-only formats keep activations bf16, so a dequantized-weight emulation is exact up to accumulation order (what a W4A16 kernel computes).

  python wq.py calib N            -> ~/work/j5/hess/H_{i}_{k}.pt  (input Hessians over N train-split real requests)
  python wq.py quant FMT          -> ~/work/j5/wq_FMT.pt          FMT: w8 | w4g128 | w4g64 | w3g128 | w4g128rtn | ...
  python wq.py eval FMT[,FMT] [suites]  -> ~/work/j5/preds_FMT.jsonl  (all evalkit questions, hobson layout, one question per sequence)
FMT 'bf16' = the unmodified fold runtime (the in-runtime floor). FMT 'mix:<a>:<b>' = format a on MLP (Wgu, Wd), b on mixers (Win, Wo).
"""
import os, sys, json, time, math, random, torch
sys.path[:0] = [os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/d1'),
                os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import triton
J5 = os.path.expanduser('~/work/j5')
HD = J5 + '/hess'
GEMMS = ('Win', 'Wo', 'Wgu', 'Wd')
KEY = {'Win': 'Win_f', 'Wo': 'Wo', 'Wgu': 'Wgu_f', 'Wd': 'Wd'}


def load():
    from kitrun import load_P
    from lean2 import Lean2
    import qrt as Q
    P = load_P()
    ln = Lean2(P.tm, fuse='fold')
    head = P.model.head.float().eval()
    m = Q.QRT(ln, head=head, prec='bf16'); m.tune = False
    return P, ln, m, Q


def cal_items(n, seed=0, minT=64, maxT=3500):
    EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])
    out = []; rng = random.Random(seed)
    with open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')) as f:
        for li, l in enumerate(f):
            if li % 89 != 7: continue
            r = json.loads(l)
            if r['task'] in EV or not minT <= r['n_state_tok'] <= maxT: continue
            qn = rng.choice(sorted(r['questions']))
            out.append((r['state'], r['questions'][qn]))
    rng.shuffle(out)
    return out[:n]


# ------------------------------------------------------------------ calibration: fold forward with input-Hessian capture
@torch.no_grad()
def calib_fwd(m, Q, ids, Hacc):
    import lean2 as L2, qk as K
    import torch.nn.functional as F
    T = ids.shape[0]; eps = m.eps
    lay = Q.Lay('single', T)
    m._ztail = torch.zeros(1, 6144, device=m.dev)
    fr = lay.pos[:, None] * m.ln2.inv[None, :]; fr = torch.cat([fr, fr], -1)
    cos, sin = fr.cos().to(torch.bfloat16).contiguous(), fr.sin().to(torch.bfloat16).contiguous()
    x = F.embedding(ids, m.ln2.embed).contiguous()
    ss = x.float().pow(2).sum(-1)

    def acc(i, k, a):
        a = a.float()
        if (i, k) not in Hacc: Hacc[(i, k)] = torch.zeros(a.shape[1], a.shape[1], device=a.device)
        Hacc[(i, k)].addmm_(a.t(), a)

    for i, d in enumerate(m.L):
        acc(i, 'Win', x.float() * torch.rsqrt(ss / 2048 + eps)[:, None])
        proj = L2.tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=2048)
        y = torch.empty(T, 2048, device=m.dev, dtype=torch.bfloat16)
        if d['type'] == 'linear_attention':
            o = m._gdn(i, d, proj, None, None, lay, None, False)
            K._gnorm_k[(triton.cdiv(T, m.rows),)](o, proj[:, 6144:], proj, proj, d['gn_w'], proj, proj, proj, proj, proj, proj, y, T, proj.stride(0), eps,
                                               DQ=False, OUTQ=False, HAD=1, QMAX=7., CLIP=1., BITS=8, ROWS=m.rows, num_warps=8)
        else:
            o = m._attn(i, d, proj, None, None, lay, None, False, cos, sin)
            K._agate_k[(triton.cdiv(T, m.rows),)](o, proj, proj, proj, proj, proj, proj, proj, proj, proj, y, T, proj.stride(0), o.stride(0), o.stride(1),
                                               DQ=False, OUTQ=False, HAD=1, QMAX=7., CLIP=1., BITS=8, ROWS=m.rows, num_warps=8)
        acc(i, 'Wo', y)
        ss = torch.zeros(T, device=m.dev, dtype=torch.float32)
        L2.tgemm(y, d['Wo'], epi=3, res=x, ssout=ss)
        acc(i, 'Wgu', x.float() * torch.rsqrt(ss / 2048 + eps)[:, None])
        mm = L2.tgemm(x, d['Wgu_f'], epi=1, ss=ss, Kd=2048)
        acc(i, 'Wd', mm)
        ss = torch.zeros(T, device=m.dev, dtype=torch.float32)
        L2.tgemm(mm, d['Wd'], epi=3, res=x, ssout=ss)


def calib(n):
    P, ln, m, Q = load()
    os.makedirs(HD, exist_ok=True)
    its = cal_items(n)
    seqs = []
    for st, qd in its:
        pr = P.prep(st, qd); ids = pr['s'] + pr['q']
        if len(ids) > 3000: ids = ids[:750] + ids[-2250:]
        seqs.append(ids)
    ntok = sum(map(len, seqs))
    print('calib seqs', len(seqs), 'tokens', ntok, 'lens', sorted(map(len, seqs))[::max(1, len(seqs) // 8)], flush=True)
    Hacc = {}; t0 = time.time()
    for j, ids in enumerate(seqs):
        calib_fwd(m, Q, torch.tensor(ids, device='cuda'), Hacc)
        if j % 16 == 0: print('seq', j, f'{time.time()-t0:.0f}s', flush=True)
    for (i, k), H in Hacc.items(): torch.save((H / ntok).cpu(), f'{HD}/H_{i}_{k}.pt')
    json.dump(dict(n=len(seqs), tokens=ntok), open(f'{HD}/meta.json', 'w'))
    print('done', f'{time.time()-t0:.0f}s', flush=True)


# ------------------------------------------------------------------ quantizers
def parse_fmt(f):
    """w8 -> (8, None, sym) per-channel; w4g128 -> (4, 128, asym); suffix 'rtn' = no GPTQ; 'sym' = symmetric groups"""
    rtn = f.endswith('rtn'); f = f.replace('rtn', '')
    sym = f.endswith('sym'); f = f.replace('sym', '')
    if 'g' in f:
        b, g = f[1:].split('g'); return int(b), int(g), sym, rtn
    return int(f[1:]), None, True, rtn


def grid_params(W, bits, g, sym):
    """per-(row, group) scale / zero with MSE clip search. W [N,K] fp32 -> s, z [N, K/g] (z=None if sym). g None = per channel."""
    N, K = W.shape; gg = g or K
    Wg = W.reshape(N, K // gg, gg)
    qmax = 2 ** bits - 1
    best_e = None; best_s = None; best_z = None
    for r in (1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79, 0.76, 0.73, 0.70):
        if sym:
            qm = 2 ** (bits - 1) - 1
            s = (Wg.abs().amax(-1) * r).clamp_min(1e-9) / qm
            q = torch.round(Wg / s[..., None]).clamp(-qm - 1, qm); deq = q * s[..., None]; z = None
        else:
            mx = Wg.amax(-1) * r; mn = Wg.amin(-1) * r
            s = ((mx - mn) / qmax).clamp_min(1e-9); z = torch.round(-mn / s).clamp(0, qmax)
            q = (torch.round(Wg / s[..., None]) + z[..., None]).clamp(0, qmax); deq = (q - z[..., None]) * s[..., None]
        e = (deq - Wg).pow(2).sum(-1)
        if best_e is None:
            best_e, best_s, best_z = e, s, z
        else:
            msk = e < best_e; best_e = torch.where(msk, e, best_e); best_s = torch.where(msk, s, best_s)
            if z is not None: best_z = torch.where(msk, z, best_z)
    return best_s, best_z


def qdq_col(w, s, z, bits, sym):
    if sym:
        qm = 2 ** (bits - 1) - 1
        q = torch.round(w / s).clamp(-qm - 1, qm); return q, q * s
    qmax = 2 ** bits - 1
    q = (torch.round(w / s) + z).clamp(0, qmax); return q, (q - z) * s


def gptq_w(W, H, bits, g, sym, blocksize=128, percdamp=0.01):
    """GPTQ, act-order, static groups (scales from the original W, so the kernel needs no g_idx). Returns codes [N,K] (float), s, z, deq."""
    W = W.clone().float(); N, K = W.shape; H = H.clone().float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    s, z = grid_params(W, bits, g, sym)
    gg = g or K
    perm = torch.argsort(torch.diag(H), descending=True)
    W = W[:, perm]; H = H[perm][:, perm]
    gidx = (perm // gg)
    damp = percdamp * torch.mean(torch.diag(H)); H[range(K), range(K)] += damp
    L = torch.linalg.cholesky(H); Hinv = torch.cholesky_inverse(L); Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Qc = torch.zeros_like(W); Dq = torch.zeros_like(W)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K); n = i2 - i1
        W1 = W[:, i1:i2].clone(); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for j in range(n):
            w = W1[:, j]; d = Hi[j, j]; gi = gidx[i1 + j]
            q, dq = qdq_col(w, s[:, gi], None if z is None else z[:, gi], bits, sym)
            Qc[:, i1 + j] = q; Dq[:, i1 + j] = dq
            e = (w - dq) / d
            W1[:, j:] -= e[:, None] * Hi[j, j:][None, :]
            E1[:, j] = e
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    inv = torch.argsort(perm)
    return Qc[:, inv], s, z, Dq[:, inv]


def rtn_w(W, bits, g, sym):
    W = W.float(); N, K = W.shape; gg = g or K
    s, z = grid_params(W, bits, g, sym)
    Wg = W.reshape(N, K // gg, gg)
    q, dq = qdq_col(Wg, s[..., None], None if z is None else z[..., None], bits, sym)
    return q.reshape(N, K), s, z, dq.reshape(N, K)


def quant(fmt):
    from kitrun import load_P
    from lean2 import Lean2
    P = load_P(); ln = Lean2(P.tm, fuse='fold')
    bits, g, sym, rtn = parse_fmt(fmt)
    out = {}; t0 = time.time(); stats = {}
    for i, d in enumerate(ln.layers):
        for k in GEMMS:
            W = d[KEY[k]].float()
            if rtn:
                q, s, z, dq = rtn_w(W, bits, g, sym)
            else:
                H = torch.load(f'{HD}/H_{i}_{k}.pt', map_location='cuda').float()
                q, s, z, dq = gptq_w(W, H, bits, g, sym)
            Hd = torch.load(f'{HD}/H_{i}_{k}.pt', map_location='cuda').float()
            E = dq - W
            # relative output error on the calibration inputs: tr(E H E^T) / tr(W H W^T)
            stats[f'{i}.{k}'] = float(((E @ Hd) * E).sum() / ((W @ Hd) * W).sum())
            out[(i, k)] = dict(q=q.to(torch.uint8 if not sym else torch.int8).cpu(), s=s.half().cpu(), z=None if z is None else z.to(torch.uint8).cpu(),
                               bits=bits, g=g, sym=sym)
            del W, E, Hd
        print('layer', i, f'{time.time()-t0:.0f}s', 'relerr', {k: round(stats[f'{i}.{k}'], 6) for k in GEMMS}, flush=True)
    torch.save(out, f'{J5}/wq_{fmt}.pt')
    json.dump(stats, open(f'{J5}/wqerr_{fmt}.json', 'w'))
    print('saved', fmt, f'{time.time()-t0:.0f}s', 'mean relerr', sum(stats.values()) / len(stats), flush=True)


def deq(e):
    q = e['q'].cuda().float(); s = e['s'].cuda().float(); N, K = q.shape; gg = e['g'] or K
    qg = q.reshape(N, K // gg, gg)
    if e['sym']: w = qg * s[..., None]
    else: w = (qg - e['z'].cuda().float()[..., None]) * s[..., None]
    return w.reshape(N, K)


def apply_fmt(ln, fmt):
    """replace the fold runtime's weights by dequantized ones. fmt: 'bf16' | FMT | 'mix:<mlp fmt>:<mixer fmt>' | 'map:<json path>'"""
    if fmt == 'bf16': return
    if fmt.startswith('mix:'):
        _, fa, fb = fmt.split(':'); per = {k: (fa if k in ('Wgu', 'Wd') else fb) for k in GEMMS}; pm = None
    elif fmt.startswith('map:'):
        pm = json.load(open(os.path.expanduser(fmt[4:]))); per = None
    else:
        per = {k: fmt for k in GEMMS}; pm = None
    cache = {}
    for i, d in enumerate(ln.layers):
        for k in GEMMS:
            f = per[k] if per else pm.get(f'{i}.{k}', pm.get('default', 'bf16'))
            if f == 'bf16': continue
            if f not in cache: cache[f] = torch.load(f'{J5}/wq_{f}.pt')
            d[KEY[k]] = deq(cache[f][(i, k)]).to(torch.bfloat16).contiguous()
    del cache; torch.cuda.empty_cache()


def evaluate(fmts, suites=None):
    import evalkit as EK
    from kitrun import load_P, prep_question, probdict
    from lean2 import Lean2
    import qrt as Q
    P = load_P(); head = P.model.head.float().eval()
    items = list(EK.all_question_items(suites))
    byid = {}
    for s in (suites or ['JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe']):
        for it in EK.load_suite(s): byid[it['id']] = it
    for fmt in fmts:
        ln = Lean2(P.tm, fuse='fold'); apply_fmt(ln, fmt)
        m = Q.QRT(ln, head=head, prec='bf16'); m.tune = False
        tagf = fmt.replace(':', '_').replace('/', '_')
        out = f'{J5}/preds_{tagf}.jsonl'; done = set()
        if os.path.exists(out):
            for l in open(out): r = json.loads(l); done.add((r['id'], r['q']))
        t0 = time.time(); n = 0
        with open(out, 'a') as f, torch.inference_mode():
            for suite, iid, qn, st, spec in items:
                if (iid, qn) in done: continue
                pr = prep_question(P, byid[iid], qn)
                ids = torch.tensor(pr['s'] + pr['q'], device='cuda'); T = ids.shape[0]
                hn = m.forward(ids, Q.Lay('single', T))
                rows = torch.tensor([T - 1] + [pr['q0'] + o for o in pr['opt']], device='cuda')
                h = m.unrot(hn[rows]).float()
                lg = (m.head(h[:1], h[1:][None]) / P.temp_for(pr['rq'].kind))[0, :pr['rq'].n_slots].float()
                pd = probdict(pr['rq'], torch.softmax(lg, -1).tolist())
                f.write(json.dumps(dict(suite=suite, id=iid, q=qn, T=T, probs=pd, logits=[round(x, 5) for x in lg.tolist()])) + '\n'); n += 1
                if n % 400 == 0: f.flush(); print(fmt, n, f'{time.time() - t0:.0f}s', flush=True)
        print('done', fmt, n, f'{time.time() - t0:.0f}s', flush=True)
        del m, ln; torch.cuda.empty_cache()


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'calib': calib(int(sys.argv[2]))
    elif cmd == 'quant':
        for f in sys.argv[2].split(','): quant(f)
    elif cmd == 'eval': evaluate(sys.argv[2].split(','), sys.argv[3].split(',') if len(sys.argv) > 3 and sys.argv[3] != 'all' else None)
