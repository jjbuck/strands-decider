"""reference hidden states from j2lib for the fused-runtime check (j2lat.py check): qa mode (full-sequence flip == the runtime's 1-question path),
gam = 0.5 on all 18 GDN layers, lam = 0.5; and causal."""
import os, sys, torch, random
sys.path[:0] = [os.path.expanduser('~/work/j2')]
from j2lib import J2, Pack
m = J2(); m.free_hf(); m.head = m.head0
g = torch.Generator().manual_seed(5); ids = torch.randint(1000, 150000, (340,), generator=g).tolist()
out = {'ids': ids}
with torch.inference_mode():
    for mode in ('causal', 'qa'):
        m.setup_bidir(mode)
        for p_ in m.gam.values(): p_.fill_(0.5)
        for p_ in m.lam.values(): p_.fill_(0.5)
        h = m.fwd_pk(Pack([(ids, 300)], m.dev, mode))
        out[mode] = h.float().cpu()
torch.save(out, os.path.expanduser('~/work/j2/latref.pt')); print('saved')
