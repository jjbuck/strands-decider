"""j5rt correctness vs h2 QRT (same layout, real-ish ids): cos of final normed hidden; plain single and packed branches."""
import os, sys, torch
sys.path[:0] = [os.path.expanduser('~/work/j5'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g')]
os.environ.setdefault('SKTUNE', '0')
import bench as HB, qrt as Q, j5rt
from lean2 import Lean2
torso, head = HB.load_model(); import j5rt; ln = j5rt.slim(Lean2(torso, fuse='fold')); del torso
bundles = torch.load(os.path.expanduser('~/work/j5/bundles.pt'))
REFP = os.environ.get('REFP', 'bf16')
ref = Q.QRT(ln, head=head, prec=REFP); ref.tune = True
def cos(a, b): a = a.float().flatten(); b = b.float().flatten(); return float(a @ b / a.norm() / b.norm())
specs = sys.argv[1].split(';')
for spec in specs:
    m = j5rt.make(ln, head, spec)
    for T in (32, 64, 128, 256, 400):
        for bn in ('jb1', 'jb4'):
            rr = HB.Req(ref, T, bundles[bn], 'plain'); rm = HB.Req(m, T, bundles[bn], 'plain'); rm.ids.copy_(rr.ids)
            with torch.inference_mode():
                pr = rr.run(); pm = rm.run()
                h0 = ref.unrot(ref.forward(rr.ids, rr.lay)); h1 = m.unrot(m.forward(rm.ids, rm.lay))
            print(spec, T, bn, 'cos', round(cos(h0, h1), 6), 'min row cos', round(float(torch.nn.functional.cosine_similarity(h0.float(), h1.float(), dim=-1).min()), 5),
                  'max|dp|', round(float((pr - pm).abs().max()), 5), flush=True)
    del m; torch.cuda.empty_cache()
