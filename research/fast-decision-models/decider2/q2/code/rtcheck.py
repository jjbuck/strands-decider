"""QRT2 sanity: q0 = T (all int4) vs H2 QRT w4a4; q0 = 0 (all int8) vs H2 QRT w8a8; mixed q0 vs a per-row splice. Same RTN codes."""
import os, sys, torch
sys.path[:0] = [os.path.expanduser('~/work/q2'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/d1'),
                os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import qrt as Q, q2rt as R
from kitrun import load_P
from lean2 import Lean2
P = load_P(); ln = Lean2(P.tm, fuse=''); head = P.model.head.float().eval()
torch.manual_seed(0)
T = 300; ids = torch.randint(1000, 100000, (T,), device='cuda'); lay = Q.Lay('single', T)
cos = lambda a, b: torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)
with torch.inference_mode():
    m2 = R.QRT2(ln, head=head, fmt='w4q8', tune=False)
    m2.q0 = T; h2_4 = m2.unrot(m2.forward(ids, lay)).float()
    m2.q0 = 0; h2_8 = m2.unrot(m2.forward(ids, lay)).float()
    m2.q0 = 200; h2_m = m2.unrot(m2.forward(ids, lay)).float()
    del m2; torch.cuda.empty_cache()
    mc = R.QRT2C(ln, head=head, fmt='w4q8', tune=False)
    mc.q0 = 200; hc_m = mc.unrot(mc.forward(ids, lay)).float()
    mc.q0 = T; hc_4 = mc.unrot(mc.forward(ids, lay)).float()
    del mc; torch.cuda.empty_cache()
    m4 = Q.QRT(ln, head=head, prec='w4a4'); m4.tune = False
    h4 = m4.unrot(m4.forward(ids, lay)).float(); del m4; torch.cuda.empty_cache()
    m8 = Q.QRT(ln, head=head, prec='w8a8'); m8.tune = False
    h8 = m8.unrot(m8.forward(ids, lay)).float(); del m8; torch.cuda.empty_cache()
    mb = Q.QRT(ln, head=head, prec='bf16')
    hb = mb.unrot(mb.forward(ids, lay)).float()
print('final hidden cos (median / min over rows):')
for nm, a, b in [('QRT2 q0=T vs H2 w4a4', h2_4, h4), ('QRT2 q0=0 vs H2 w8a8', h2_8, h8), ('H2 w4a4 vs bf16', h4, hb), ('H2 w8a8 vs bf16', h8, hb),
                 ('QRT2 q0=T vs bf16', h2_4, hb), ('QRT2 q0=0 vs bf16', h2_8, hb), ('QRT2 q0=200 vs bf16 (state rows)', h2_m[:200], hb[:200]),
                 ('QRT2 q0=200 vs bf16 (question rows)', h2_m[200:], hb[200:]), ('QRT2 q0=200 vs q0=T (state rows, causal => equal)', h2_m[:200], h2_4[:200]),
                 ('QRT2C q0=T vs H2 w4a4 (should be exact)', hc_4, h4), ('QRT2C q0=200 vs QRT2 q0=200 (question rows)', hc_m[200:], h2_m[200:]),
                 ('QRT2C q0=200 vs bf16 (question rows)', hc_m[200:], hb[200:])]:
    c = cos(a, b); print(f'  {nm:52s} {c.median().item():.5f} / {c.min().item():.5f}')
