"""layer-range execution check: k64rr QRT2C full forward == [0,16) then [16,24) from the saved residual; taps at 16 == residual after 16"""
import os, sys, torch
sys.path[:0] = [os.path.expanduser('~/work/q2'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/d1'),
                os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens'), os.path.expanduser('~/work/j15')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch.nn.functional as F
import qrt as Q, q2rt as R
from q2run import build
P, m, head = build('c:map:' + os.path.expanduser('~/work/q2/q2map_k64rr.json'))
T = 700; ids = torch.randint(1000, 100000, (T,), device='cuda'); lay = Q.Lay('single', T); m.q0 = 600
rows = torch.tensor([T - 1, 650, 660], device='cuda')
with torch.inference_mode():
    m.taps = (16,); m.taprows = rows; m.tapped = {}
    hn = m.forward(ids, lay).clone(); x16 = m.tapped[16].clone()
    m.taps = ()
    x = F.embedding(ids, m.embed).contiguous()
    m.forward(ids, lay, x0=x, i0=0, i1=16); a = x[rows].clone()
    hn2 = m.forward(ids, lay, x0=x, i0=16, i1=24)
print('tap == residual after 16:', torch.equal(a, x16), ' split forward == full:', torch.equal(hn2, hn), (hn2.float() - hn.float()).abs().max().item())
