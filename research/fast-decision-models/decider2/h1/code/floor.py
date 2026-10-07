"""is W8A8 error above the bf16 noise floor?  KL(dense || X) per item for X = fp32-path-no-quant (one GEMM / all), W8A8 all, W8A8 one GEMM."""
import os, sys, json, time
sys.path[:0] = [os.path.expanduser('~/work/h1'), os.path.expanduser('~/work/g2')]
import torch, h1lib as HL
g = HL.H1()
its = HL.cal_items(40, seed=1); data = []
for st, qd in its:
    pr = g.prep(st, qd); ids = pr['s'] + pr['q']
    if len(ids) <= 3000: data.append((pr, ids))
cf = {'one_fp32_0.Win': {(0, 'Win'): 'w16a16'}, 'one_fp32_12.Wgu': {(12, 'Wgu'): 'w16a16'}, 'all_fp32': g.uniform('w16a16'), 'all_w8a8': g.uniform('w8a8'),
      'one_w8_12.Wgu': {(12, 'Wgu'): 'w8a8'}, 'all_w8a16': g.uniform('w8a16'), 'all_w16a8': g.uniform('w16a8')}
out = {c: [] for c in cf}; hid = {c: [] for c in cf}
for pr, ids in data:
    g.set_prec({}); h0, _ = g.fwd(ids); l0 = g.logits(h0, pr); p0 = torch.log_softmax(l0, -1)
    rows = torch.tensor([len(ids) - 1] + [pr['q0'] + o for o in pr['opt']], device=g.dev)
    for c, P in cf.items():
        g.set_prec(P); h1, _ = g.fwd(ids); l1 = g.logits(h1, pr); p1 = torch.log_softmax(l1, -1)
        out[c].append(round(float((p0.exp() * (p0 - p1)).sum()), 7))
        hid[c].append(round(float((h1[rows].float() - h0[rows].float()).norm() / h0[rows].float().norm()), 5))
for c in cf: print(c, 'meanKL', sum(out[c]) / len(out[c]), 'mean hid relerr', sum(hid[c]) / len(hid[c]), '\n   KL', out[c][:12], '\n   hid', hid[c][:12], flush=True)
