"""layer-by-layer divergence: H2 runtime (first L layers) vs H1 emulation (stop=L) on the same real ids; cos of the normed residual."""
import sys, os, json, copy, torch
sys.path[:0] = [os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/g2'), os.path.expanduser('~/work/h1'), os.path.expanduser('~/work/evalkit'),
                os.path.expanduser('~/work/tokens'), os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/d1')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import h1lib as H, evalkit as EK, qrt as Q
from kitrun import prep_question
from lean2 import Lean2
h = H.H1()
ln = Lean2(h.p.tm, fuse='fold')
its = EK.load_suite('REAL-agree')[:6]
res = {}
for prec in sys.argv[1].split(','):
    h.set_prec(h.uniform(prec)); h.drop_cache()
    for L in (1, 2, 4, 8, 16, 24):
        ln2 = copy.copy(ln); ln2.layers = ln.layers[:L]
        m = Q.QRT(ln2, prec=prec); m.tune = False
        cs = []
        for it in its:
            q = sorted(it['questions'])[0]
            pr = prep_question(h.p, it, q); ids = pr['s'] + pr['q']
            with torch.no_grad():
                _, caps = h.fwd(ids, keep_x=(L - 1,), stop=L)
                xr = caps[L - 1].float(); xr = xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + 1e-6)
                hn = m.forward(torch.tensor(ids, device='cuda'), Q.Lay('single', len(ids)))
                mine = m.R1.inv(hn.float())
            cs.append(torch.nn.functional.cosine_similarity(mine, xr, dim=-1))
        c = torch.cat(cs)
        res[f'{prec}|{L}'] = dict(cos_mean=c.mean().item(), cos_p01=c.quantile(0.01).item(), cos_min=c.min().item())
        print(prec, L, res[f'{prec}|{L}'], flush=True)
        del m; torch.cuda.empty_cache()
json.dump(res, open(os.path.expanduser('~/work/h2/h1layer.json'), 'w'), indent=1)
