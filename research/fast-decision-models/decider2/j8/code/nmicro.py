"""nmicro.py (inf2): compile + time single blocks to find where Neuron time goes.
  python nmicro.py PART T [C] [CAST]   PART in mlp | gdn | gdnmix (GDN mixer only, no MLP) | attn | gemm
"""
import os, sys, json, time, statistics as st
sys.path.insert(0, os.path.expanduser('~/work/j8'))
import torch, torch.nn as nn, torch.nn.functional as F
import torch_neuronx
import hob

part, T = sys.argv[1], int(sys.argv[2]); C = int(sys.argv[3]) if len(sys.argv) > 3 else 64
cast = sys.argv[4] if len(sys.argv) > 4 else 'none'
W, _, _ = hob.load_weights()
keep = {k: v for k, v in W.items() if k.startswith('layers.0.') or k.startswith('layers.3.') or not k.startswith('layers.')}
for i in range(24):  # Hob needs all layers; give the unused ones layer-0/3 copies (not used by the block modules)
    src = 0 if i not in (3, 7, 11, 15, 19, 23) else 3
    for k, v in W.items():
        if k.startswith(f'layers.{src}.'): keep[k.replace(f'layers.{src}.', f'layers.{i}.', 1)] = v
h = hob.Hob(keep, dtype=torch.bfloat16, C=C, attn='explicit').eval(); del W, keep


class Blk(nn.Module):
    def __init__(s):
        super().__init__(); s.h = h; s.m0 = h.L[0]; s.m3 = h.L[3]
        s.WguT = nn.Parameter(h.L[0].Wgu.t().contiguous(), requires_grad=False); s.WdT = nn.Parameter(h.L[0].Wd.t().contiguous(), requires_grad=False)
    def forward(s, x):
        hh = s.h; Tn = x.shape[0]
        if part == 'mlp':
            return x + hh.mlp(s.m0, x)
        if part == 'gemm':
            return x @ s.m0.Wgu.t()
        if part == 'gemmT':
            return x @ s.WguT
        if part == 'mlpT':
            h2 = hob.rms_zc(x, s.m0.post1); gu = h2 @ s.WguT
            return x + (F.silu(gu[:, :6144]) * gu[:, 6144:]) @ s.WdT
        if part in ('gdn', 'gdnmix'):
            m = s.m0
            hn = hob.rms_zc(x, m.in1); proj = hn @ m.Win.t()
            z, beta, g = hh.gdn_prep(m, proj)
            c = hh.conv(proj[None, :, :6144], m.conv)
            q, k, v = hh.qkv_heads(c, 1, Tn)
            o, _ = hh.chunk(q, k, v, g.t()[None], beta.t()[None])
            o = hh.gated_norm(o[0].transpose(0, 1), z, m.gnw)
            x = x + o @ m.Wo.t()
            return x if part == 'gdnmix' else x + hh.mlp(m, x)
        if part == 'attn':
            m = s.m3
            cos, sin = hh.rope_tab(torch.arange(Tn))
            hn = hob.rms_zc(x, m.in1); proj = hn @ m.Win.t()
            q, k, v, gate = hh.attn_prep(m, proj, cos, sin)
            o = hh.sdpa(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], causal=True)
            o = o[0].transpose(0, 1).reshape(Tn, 2048) * gate
            return x + o @ m.Wo.t()


mod = Blk()
x = (torch.randn(T, 2048) * 0.5).bfloat16()
args = ['--model-type', 'transformer', '--auto-cast', cast] + (['--auto-cast-type', 'bf16'] if cast != 'none' else [])
t0 = time.time()
tr = torch_neuronx.trace(mod, (x,), compiler_args=args, compiler_workdir=os.path.expanduser(f'~/work/j8/neff/wdm_{part}_{T}_{C}_{cast}'))
tc = time.time() - t0
for _ in range(5): tr(x)
ts = []
for _ in range(20):
    t = time.perf_counter(); tr(x); ts.append((time.perf_counter() - t) * 1000)
with torch.inference_mode():
    ref = mod(x.clone()).float()
err = float((tr(x).float() - ref).abs().max() / ref.abs().max())
out = dict(part=part, T=T, C=C, cast=cast, compile_s=round(tc), median_ms=round(st.median(ts), 3), min_ms=round(min(ts), 3), rel_err=err)
print(json.dumps(out), flush=True)
open(os.path.expanduser('~/work/j8/nmicro.jsonl'), 'a').write(json.dumps(out) + '\n')
