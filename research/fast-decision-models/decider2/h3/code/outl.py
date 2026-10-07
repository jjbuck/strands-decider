"""H3: activation outlier statistics at every GEMM input of hobson-v19 (why per-token int4 fails), plus norm-gain stats.
Per GEMM input and layer: median over tokens of crest = max|x_t| / rms(x_t); fraction of entries that per-token int4 rounds to 0 (|x| < amax/14);
per-token int4 relative quantization error ||Q(x)-x||/||x||; the same after the G2 Hadamard rotation; with/without the norm gain (fold).
python outl.py --n 8"""
import os, sys, json, argparse
sys.path.insert(0, os.path.expanduser('~/work/h3'))
import torch
import h3lib as H
from errprop_samples import samples

ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=8); a = ap.parse_args()
m = H.H3(); m.free_hf()
S = samples(a.n, 700, 1600)
acc = {}


def stats(x):
    x = x.float()
    rms = x.pow(2).mean(-1).sqrt().clamp_min(1e-8); amax = x.abs().amax(-1)
    crest = (amax / rms).median().item()
    zero = (x.abs() < amax[:, None] / 14).float().mean().item()
    e4 = ((H.q_rows(x, 4) - x).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-8)).mean().item()
    nv = ((H.q_nvfp4(x) - x).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-8)).mean().item()
    return [crest, zero, e4, nv]


def put(i, k, v):
    e = acc.setdefault(f'{i}', {}).setdefault(k, [0.0] * len(v) + [0])
    for j, z in enumerate(v): e[j] += z
    e[-1] += 1


with torch.inference_mode():
    for rid, qn, st, qd in S:
        pr = m.prep(st, qd); ids = pr['s'] + pr['q']
        cur = {}

        def cap(i, pos, t):
            if pos == 'layer_in':
                xn = H.nrm(t, m.eps); put(i, 'Win_in_nogain', stats(xn)); put(i, 'Win_in_nogain_rot', stats(H.rot_rows(xn)))
            if pos == 'attn_in': put(i, 'Win_in', stats(t))
            if pos == 'o_in': put(i, 'Wo_in', stats(t)); put(i, 'Wo_in_rot', stats(H.rot_rows(t)))
            if pos == 'A+R':
                xn = H.nrm(t, m.eps); put(i, 'Wgu_in_nogain', stats(xn)); put(i, 'Wgu_in_nogain_rot', stats(H.rot_rows(xn)))
            if pos == 'mlp_in': put(i, 'Wgu_in', stats(t))
            if pos == 'mlp_hid': put(i, 'Wd_in', stats(t)); put(i, 'Wd_in_rot', stats(H.rot_rows(t)))
        m.forward(ids, H.DENSE, cap=cap)
out = {i: {k: dict(crest=v[0] / v[-1], zero_frac=v[1] / v[-1], int4_rel=v[2] / v[-1], nvfp4_rel=v[3] / v[-1]) for k, v in d.items()} for i, d in acc.items()}
gains = {i: dict(in_max=float(m.L[i]['in1'].max()), in_min=float(m.L[i]['in1'].min()), in_mean=float(m.L[i]['in1'].mean()),
                 post_max=float(m.L[i]['post1'].max()), post_mean=float(m.L[i]['post1'].mean())) for i in range(24)}
wst = {}
for i in range(24):
    for nm in H.NAMES:
        Wf = m.L[i][nm].float()
        e4 = ((H.q_rows_mse(Wf, 4) - Wf).norm() / Wf.norm()).item(); nv = ((H.q_nvfp4(Wf) - Wf).norm() / Wf.norm()).item()
        wst[f'{i}.{nm}'] = dict(int4_rel=e4, nvfp4_rel=nv, row_crest=float((Wf.abs().amax(1) / Wf.pow(2).mean(1).sqrt()).median()))
json.dump(dict(act=out, gains=gains, w=wst), open(os.path.expanduser('~/work/h3/res/outl.json'), 'w'))
print('done')
