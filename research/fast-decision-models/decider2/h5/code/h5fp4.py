"""H5: NVFP4 W4A4 (next-generation hardware: Blackwell / RTX 5090 block-scaled FP4) emulated on the A10G, RTN weights, on the train-split dev set.
Variants: no rotation at all; Hadamard (R1/Ho/Hd); learned R1. python h5fp4.py"""
import os, sys, json, time, random
sys.path.insert(0, os.path.expanduser('~/work/h5'))
import torch
import h5lib as H
W = os.path.expanduser('~/work/h5/')
m = H.Q5(lean=True); m.aclip = 1.0
CL = ('Win_g', 'Win_a', 'Wo_g', 'Wo_a', 'Wgu', 'Wd')
pool = H.load_pool(150, 3000)
rng = random.Random(11); dev_pool = list(pool); rng.shuffle(dev_pool)
DEV = []
for r in dev_pool[:240]:
    qn = rng.choice(sorted(r['questions'])); pr = m.prep(r['state'], r['questions'][qn]); DEV.append((pr, pr['s'] + pr['q']))
REF = torch.load(W + 'res/devref_240.pt')
R_had, Ho, Hd = m.R1.clone(), m.Ho, m.Hd
R_l1 = torch.load(W + 'rot/R1_l1.pt').to(m.dev).float()
out = {}
for name, fmt, rot in [('nvfp4_norot', 'nvfp4', 'none'), ('nvfp4_had', 'nvfp4', 'had'), ('nvfp4_l1', 'nvfp4', 'l1'), ('nvfp4_A_only_had', 'nvfp4a', 'had'),
                       ('nvfp4_W_only_had', 'nvfp4w', 'had')]:
    t0 = time.time()
    if rot == 'none': m.R1 = torch.eye(2048, device=m.dev); m.Ho = H.Ident(); m.Hd = H.Ident()
    else: m.R1 = R_had if rot == 'had' else R_l1; m.Ho = Ho; m.Hd = Hd
    H.FMT['a'] = 'nvfp4' if fmt in ('nvfp4', 'nvfp4a') else 'int'; H.FMT['w'] = 'nvfp4' if fmt in ('nvfp4', 'nvfp4w') else 'int'
    if fmt == 'nvfp4a': m.cfg = {c: (16, 4) for c in CL}
    elif fmt == 'nvfp4w': m.cfg = {c: (4, 16) for c in CL}
    else: m.cfg = {c: (4, 4) for c in CL}
    m.build_rtn()
    fl = 0; tv = 0.0; kl = 0.0
    with torch.no_grad():
        for (pr, ids), r in zip(DEV, REF):
            h, _ = m.forward(ids, 'q'); p = torch.softmax(m.logits(h, pr).float(), -1).cpu()
            fl += int(p.argmax() != r.argmax()); tv += H.tv(p, r); kl += H.kl(r, p)
    n = len(DEV); out[name] = dict(flips=fl, n=n, flip_rate=round(fl / n, 4), tv=round(tv / n, 4), kl=round(kl / n, 4), s=round(time.time() - t0))
    print(name, json.dumps(out[name]), flush=True)
    json.dump(out, open(W + 'res/fp4.json', 'w'), indent=1)
