"""where does W4A4 differ from H1's emulation? layer 0, Win: activation codes, weight codes, GEMM output."""
import sys, os, copy, torch
sys.path[:0] = [os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/g2'), os.path.expanduser('~/work/h1'), os.path.expanduser('~/work/evalkit'),
                os.path.expanduser('~/work/tokens'), os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/d1')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import h1lib as H, evalkit as EK, qrt as Q, qk as K, qgemm as QG
from kitrun import prep_question
from lean2 import Lean2
h = H.H1()
ln = Lean2(h.p.tm, fuse='fold'); ln2 = copy.copy(ln); ln2.layers = ln.layers[:2]
def unpack4(b):
    lo = (b & 15).to(torch.int16); hi = ((b >> 4) & 15).to(torch.int16)
    lo = torch.where(lo >= 8, lo - 16, lo); hi = torch.where(hi >= 8, hi - 16, hi)
    return torch.stack([lo, hi], -1).reshape(b.shape[0], -1)
it = EK.load_suite('REAL-agree')[0]; q = sorted(it['questions'])[0]
pr = prep_question(h.p, it, q); ids = torch.tensor(pr['s'] + pr['q'], device='cuda'); T = ids.shape[0]
for prec, bits in (('w8a8', 8), ('w4a4', 4)):
    h.set_prec(h.uniform(prec)); h.drop_cache()
    m = Q.QRT(ln2, prec=prec); m.tune = False
    x = torch.nn.functional.embedding(ids, h.embed)
    xf = x.float(); xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + h.eps)
    xr = h.rot_for(0, 'Win')(xn); qa, sa = h.qact(xr, bits); qw, sw = h.qweight(0, 'Win', prec)
    yH = h.imm(qa, qw) * sa[:, None] * sw[None, :]
    # mine
    xt = torch.nn.functional.embedding(ids, m.embed).contiguous()
    nxt = m._next_in(0, 'Win', T)
    K.addq(xt, None, None, None, nxt['q'], nxt['s'], nxt['hb'], h.eps, dq=False, outq=True, qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'])
    qm = unpack4(nxt['q']) if bits == 4 else nxt['q'].to(torch.int16)
    e = m.qw[0]['Win']; wm = unpack4(e['codes']) if bits == 4 else e['codes'].to(torch.int16)
    print(prec, 'act code mismatch', (qm.float() != qa.float()).float().mean().item(), 'act scale rel', ((nxt['s'] - sa).abs() / sa).max().item(), flush=True)
    print(prec, 'weight code mismatch', (wm.float() != qw.float()).float().mean().item(), 'w scale rel', ((e['csa'] * e['alpha'] - sw).abs() / sw).max().item(), flush=True)
    P16 = m.qgemm(nxt['q'], e); ym = P16.float() * nxt['s'][:, None] * e['csa'][None, :]
    print(prec, 'Win output rel err', ((ym - yH).norm() / yH.norm()).item(), flush=True)
    # same codes through my GEMM
    yq = (QG.gemm(e['kind'], QG.pack4(qa) if bits == 4 else qa.contiguous(), e['codes'], e['alpha'], 3).float() * sa[:, None] * e['csa'][None, :])
    print(prec, 'Win output with H1 act codes rel err', ((yq - yH).norm() / yH.norm()).item(), flush=True)
    # residual-basis bf16 effect
    print(prec, 'embed rotated vs H1 rotated-normed: cos', torch.nn.functional.cosine_similarity(xt.float(), m.R1(x.float()), dim=-1).mean().item())
    del m; torch.cuda.empty_cache()
