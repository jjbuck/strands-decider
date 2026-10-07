"""Q3 init: H1's GPTQ recipe in the H1/H2 rotated W4 format, plus GPTQ's error-compensated weights (the latent / AdaRound start point).
1. Hessians: unrotated GEMM-input second moments from 64 train-split (state, question) pairs (h1lib.cal_items(64, seed=0), 3000-token cap,
   same selection as H1/H2/J15), computed with the bf16 teacher (gain-free normed residual for Win/Wgu, mixer output for Wo, silu(g)*u for Wd).
2. GPTQ act-order, damp .01, block 128, MSE-clip per-channel scale (h1lib.gptq), on R^T H R of the rotated folded weight.
Writes ~/work/q3/gptq4.pt: {(i,k): dict(q int8 [N,K], s fp32 [N], Wc bf16 [N,K])}, with round(Wc/s) == q exactly.
python q3gptq.py [--n 64]"""
import os, sys, time, json, random, argparse
sys.path[:0] = [os.path.expanduser('~/work/q3')]
import torch, torch.nn.functional as F
import q3lib as QL

ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=64); ap.add_argument('--maxtok', type=int, default=3000)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', default=''); ap.add_argument('--bits', type=int, default=4); a = ap.parse_args()
if not a.out: a.out = f'~/work/q3/gptq{a.bits}.pt'
QM = 7.0 if a.bits == 4 else 127.0
W = os.path.expanduser('~/work')


def cal_items(n, seed=0, minT=300, maxT=3500, skip=0):
    """= h1lib.cal_items"""
    EV = set(json.load(open(f'{W}/evalkit/split.json'))['eval_tasks'])
    out = []; rng = random.Random(seed)
    with open(f'{W}/evalkit/train_pool.jsonl') as f:
        for li, l in enumerate(f):
            if li % 97 != 13 + seed: continue
            r = json.loads(l)
            if r['task'] in EV or not minT <= r['n_state_tok'] <= maxT: continue
            qn = rng.choice(sorted(r['questions']))
            out.append((r['state'], r['questions'][qn]))
    return out[skip:skip + n]


m = QL.Q3(); dev = m.dev; eps = m.eps
its = cal_items(a.n, a.seed)
seqs = []
for st, qd in its:
    pr = m.p.prep(st, qd); ids = pr['s'] + pr['q']
    if len(ids) > a.maxtok: ids = ids[:a.maxtok // 4] + ids[-(a.maxtok - a.maxtok // 4):]
    seqs.append(ids)
ntok = sum(map(len, seqs)); print('calib seqs', len(seqs), 'tokens', ntok, flush=True)
Hs = {}


def acc(key, x):
    xs = x.float(); u = xs.t() @ xs
    Hs[key] = u if key not in Hs else Hs[key] + u


t0 = time.time()
with torch.no_grad():
    for ids in seqs:
        rq = dict(s=ids, qs=[])
        lay = m.layout(rq)
        x = F.embedding(lay['ids'], m.embed)
        for i in range(24):
            d = m.L[i]
            acc((i, 'Win'), QL.nrm32(x, eps))
            proj = QL.rms_zc(x, d['in_norm'], eps) @ d['Win'].t()
            o = m._gdn(i, d, proj, lay) if d['type'] == 'linear_attention' else m._attn(i, d, proj, lay)
            acc((i, 'Wo'), o)
            x = x + o @ d['Wo'].t()
            acc((i, 'Wgu'), QL.nrm32(x, eps))
            gu = QL.rms_zc(x, d['post_norm'], eps) @ d['Wgu'].t(); I = d['I']
            mm = F.silu(gu[:, :I]) * gu[:, I:]
            acc((i, 'Wd'), mm)
            x = x + mm @ d['Wd'].t()
print('hessians', f'{time.time() - t0:.0f}s', f'mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB', flush=True)
out = {}; stats = []
for i in range(24):
    for k in QL.GEMMS:
        H0 = Hs.pop((i, k)) / ntok
        R = m.R4 if k == 'Wd' else (m.R2 if k == 'Wo' else m.R1)
        Hm = R(R(H0).t().contiguous())
        Wf = m.wfold(i, k)
        q, s, Wc = QL.gptq_c(Wf, Hm, QM)
        if a.bits == 8:
            out[(i, k)] = dict(q=q.to(torch.int8).cpu(), s=s.float().cpu()); stats.append(dict(i=i, k=k)); del H0, Hm, Wf, q, Wc; continue
        Wb = Wc.to(torch.bfloat16)
        bad = torch.round(Wb.float() / s[:, None]).clamp(-7, 7) != q
        nb = int(bad.sum())
        if nb:
            Wb[bad] = (q[bad] * s[:, None].expand_as(q)[bad]).to(torch.bfloat16)
        assert torch.equal(torch.round(Wb.float() / s[:, None]).clamp(-7, 7), q)
        # proxy: relative output error of GPTQ vs RTN on the calibration second moment, tr(E H E^T) / tr(W H W^T)
        def rel(Wq):
            E = (Wq - Wf); return float((E @ Hm * E).sum() / (Wf @ Hm * Wf).sum())
        srtn = QL.rtn_scales(Wf, 7.0, True); qr = torch.round(Wf / srtn[:, None]).clamp(-7, 7)
        stats.append(dict(i=i, k=k, fixed=nb, rel_gptq=rel(q * s[:, None]), rel_rtn=rel(qr * srtn[:, None]), rel_lat=rel(Wb.float())))
        out[(i, k)] = dict(q=q.to(torch.int8).cpu(), s=s.float().cpu(), Wc=Wb.cpu())
        del H0, Hm, Wf, q, Wc, Wb, qr
    print('layer', i, f'{time.time() - t0:.0f}s', json.dumps(stats[-4:]), flush=True)
torch.save(out, os.path.expanduser(a.out))
json.dump(stats, open(os.path.expanduser(a.out).replace('.pt', '_stats.json'), 'w'))
print('saved', f'{time.time() - t0:.0f}s', flush=True)
