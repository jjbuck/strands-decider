"""Fast format screen on train-split DEV requests (tau tasks disjoint from the spectrum set A): dense and quantized forwards only.
Per config: flips vs dense, rms of the margin change dm = m_q - m_dense (m = top1 - top2 logit of the dense run), mean TV, and the
predicted flip count sum_r Phi(-m0_r / rms_dm) (validated on W4A4: 24.0 predicted vs 26 actual of 300).
python q1dm.py --tags a,b,c --n 300 -> ~/work/q1/dm/{tag}.json (per-request m0, dm, flip, tv) and prints a line per config."""
import os, sys, json, time, argparse, math, zlib
sys.path[:0] = [os.path.expanduser('~/work/q1')]
import torch
import q1lib as QL, q1fmt as QF

ap = argparse.ArgumentParser(); ap.add_argument('--tags', required=True); ap.add_argument('--n', type=int, default=300)
ap.add_argument('--out', default=os.path.expanduser('~/work/q1/dm')); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
CF = json.load(open(os.path.expanduser('~/work/q1/cfgs.json')))
g = QL.Q1(grad=False)
_, B = QL.req_sets()
items = QL.dev_set(a.n, skip_rids=[b['rid'] for b in B])
P = []
for it in items:
    pr = g.prep(it['state'], it['q']); P.append((it, pr, pr['s'] + pr['q']))
dense_f = f'{a.out}/_dense.pt'
if os.path.exists(dense_f): D = torch.load(dense_f)
else:
    g.qfn = {}; D = []
    with torch.no_grad():
        for it, pr, ids in P:
            h, _ = g.fwd(ids, q0=pr['q0']); D.append(g.logits(h, pr).cpu())
    torch.save(D, dense_f)
Phi = lambda z: 0.5 * math.erfc(-z / math.sqrt(2))
for tag in a.tags.split(','):
    if os.path.exists(f'{a.out}/{tag}.json'): print('have', tag); continue
    fm = QF.Fmt(g, CF[tag]); fm.install(g); t0 = time.time(); rows = []
    with torch.no_grad():
        for (it, pr, ids), l0 in zip(P, D):
            fm.seed = zlib.crc32(it['rid'].encode())
            h, _ = g.fwd(ids, q0=pr['q0']); lq = g.logits(h, pr).cpu()
            o = torch.argsort(l0, descending=True); t1, t2 = int(o[0]), int(o[1])
            m0 = float(l0[t1] - l0[t2]); dm = float(lq[t1] - lq[t2]) - m0
            p0 = torch.softmax(l0, -1); pq = torch.softmax(lq, -1)
            rows.append(dict(rid=it['rid'], T=len(ids), q0=pr['q0'], m0=m0, dm=dm, flip=int(torch.argmax(lq)) != t1, tv=0.5 * float((pq - p0).abs().sum())))
    g.qfn = {}
    n = len(rows); rms = math.sqrt(sum(r['dm'] ** 2 for r in rows) / n)
    res = dict(tag=tag, spec=CF[tag], work=fm.work(), n=n, flips=sum(r['flip'] for r in rows), rms_dm=rms, tv=sum(r['tv'] for r in rows) / n,
               pred_flips=sum(Phi(-r['m0'] / max(rms, 1e-9)) for r in rows), sec=time.time() - t0, rows=rows)
    json.dump(res, open(f'{a.out}/{tag}.json', 'w'))
    w = res['work']['s']
    print(f"{tag}: flips {res['flips']}/{n} rms_dm {rms:.4f} TV {res['tv']:.4f} pred_flips {res['pred_flips']:.1f} | state rows int4 {w['int4']:.3f} int8 {w['int8']:.3f} "
          f"bf16 {w['bf16']:.3f} extra {w['extra_bf16_macs']:.3f} | {res['sec']:.0f}s", flush=True)
    del fm; torch.cuda.empty_cache()
