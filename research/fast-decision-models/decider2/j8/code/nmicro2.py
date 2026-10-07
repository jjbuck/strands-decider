"""nmicro2.py (inf2): localize GDN cost. PART: prep (proj+conv+l2norm, current layout) | core (chunk rule from q,k,v) ; env HOB_SCAN, HOB_TRI.
  python nmicro2.py PART T C CAST"""
import os, sys, json, time, statistics as st
sys.path.insert(0, os.path.expanduser('~/work/j8'))
import torch, torch.nn as nn
import torch_neuronx
import hob

part, T, C, cast = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
torch.manual_seed(0)
if part == 'core':
    class M(nn.Module):
        def forward(s, q, k, v, g, b):
            o, S = hob.gdn_chunk(hob.l2n(q) * 128 ** -0.5, hob.l2n(k), v, g, b, C)
            return o
    ex = (torch.randn(1, 16, T, 128), torch.randn(1, 16, T, 128), torch.randn(1, 16, T, 128), -torch.rand(1, 16, T) * 0.5, torch.rand(1, 16, T))
    mod = M()
else:
    from safetensors import safe_open
    _, bc = hob.ckpt_dirs()
    import glob
    f = safe_open(glob.glob(f'{bc}/model.safetensors*.safetensors')[0], 'pt')
    W = {k[len('model.language_model.'):]: f.get_tensor(k) for k in f.keys() if k.startswith('model.language_model.layers.0.')}
    class M(nn.Module):
        def __init__(s):
            super().__init__()
            s.Win = nn.Parameter(torch.cat([W['layers.0.linear_attn.in_proj_qkv.weight'], W['layers.0.linear_attn.in_proj_z.weight'], W['layers.0.linear_attn.in_proj_b.weight'], W['layers.0.linear_attn.in_proj_a.weight']], 0).bfloat16(), requires_grad=False)
            s.cw = nn.Parameter(W['layers.0.linear_attn.conv1d.weight'].squeeze(1).float(), requires_grad=False)
        def forward(s, x):
            proj = x @ s.Win.t()
            h = hob.Hob.__new__(hob.Hob); h.dt = torch.bfloat16; h.gdt = torch.float32
            c = hob.Hob.conv(h, proj[None, :, :6144], s.cw)
            q, k, v = hob.Hob.qkv_heads(h, c, 1, x.shape[0])
            return q + k + v
    ex = ((torch.randn(T, 2048) * 0.5).bfloat16(),)
    mod = M()
args = ['--model-type', 'transformer', '--auto-cast', cast] + (['--auto-cast-type', 'bf16'] if cast != 'none' else [])
tag = f"{part}_{T}_{C}_{cast}_scan{int(hob.SCAN)}"
t0 = time.time()
tr = torch_neuronx.trace(mod, ex, compiler_args=args, compiler_workdir=os.path.expanduser(f'~/work/j8/neff/wdm2_{tag}'))
tc = time.time() - t0
for _ in range(5): tr(*ex)
ts = []
for _ in range(20):
    t = time.perf_counter(); tr(*ex); ts.append((time.perf_counter() - t) * 1000)
with torch.inference_mode():
    ref = mod(*ex).float()
err = float((tr(*ex).float() - ref).abs().max() / ref.abs().max())
out = dict(part=part, T=T, C=C, cast=cast, scan=hob.SCAN, compile_s=round(tc), median_ms=round(st.median(ts), 3), min_ms=round(min(ts), 3), rel_err=err)
print(json.dumps(out), flush=True)
open(os.path.expanduser('~/work/j8/nmicro.jsonl'), 'a').write(json.dumps(out) + '\n')
