import sys, os, torch
sys.path[:0] = [os.path.expanduser('~/work/h2')]
import bench as B, qrt as Q
from lean2 import Lean2
bundles = torch.load(os.path.expanduser('~/work/h2/bundles.pt'))
torso, head = B.load_model(); ln = Lean2(torso, fuse='fold'); del torso
for prec in ('bf16', 'w8a8'):
    m = Q.QRT(ln, head=head, prec=prec)
    for md in ('plain', 'schema'):
        req = B.Req(m, 1000, bundles['4q'], md)
        with torch.inference_mode():
            p1 = req.run(); p2 = req.run()
        print(prec, md, 'rows', req.lay.T, 'P', req.P, 'repeat-stable', torch.equal(p1, p2), [round(x, 4) for x in p1[:, :3].flatten().tolist()], flush=True)
