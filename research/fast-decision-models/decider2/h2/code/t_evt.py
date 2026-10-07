import sys, os, torch
sys.path[:0] = [os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g')]
from prof_d1 import load_torso
from lean2 import Lean2
import qrt as Q
torch.manual_seed(0)
torso = load_torso(); ln = Lean2(torso, fuse='fold'); del torso
ids = torch.randint(1000, 100000, (1000,), device='cuda')
for prec in ('w4a4', 'w8a8', 'w4a8'):
    os.environ['EVT'] = '0'; m0 = Q.QRT(ln, prec=prec); h0 = m0.unrot(m0.forward(ids, Q.Lay('single', 1000))).float(); del m0
    os.environ['EVT'] = '1'; m1 = Q.QRT(ln, prec=prec); h1 = m1.unrot(m1.forward(ids, Q.Lay('single', 1000))).float()
    c = torch.nn.functional.cosine_similarity(h0, h1, dim=-1)
    print(prec, 'EVT vs non-EVT final hidden cos mean', round(c.mean().item(), 5), 'min', round(c.min().item(), 4), 'evt used:', m1.qw[0]['Wgu'].get('evt', False), flush=True)
    del m1; torch.cuda.empty_cache()
