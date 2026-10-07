"""nbuild.py (inf2): trace hob.py with torch_neuronx for one exact shape; save the NEFF-backed TorchScript module.
  python nbuild.py --T 1000 --M 1            single question: ids [T+Lq], sel [K+1]
  python nbuild.py --T 1000 --M 4            state once + 4 question branches (packed)
  python nbuild.py --L 1024 --sel 32 --tag evalL1024   eval bucket: right-padded ids [L], sel [32]
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.expanduser('~/work/j8'))
import torch, torch.nn as nn
import torch_neuronx
import hob

ap = argparse.ArgumentParser()
ap.add_argument('--T', type=int, default=0); ap.add_argument('--M', type=int, default=1)
ap.add_argument('--L', type=int, default=0); ap.add_argument('--sel', type=int, default=32)
ap.add_argument('--C', type=int, default=64); ap.add_argument('--cast', default='none')
ap.add_argument('--opt', default=''); ap.add_argument('--tag', default='')
a = ap.parse_args()
OUT = os.path.expanduser('~/work/j8/neff'); os.makedirs(OUT, exist_ok=True)
LI = json.load(open(os.path.expanduser('~/work/j8/lat_inputs.json')))


class Single(nn.Module):
    def __init__(s, h): super().__init__(); s.h = h
    def forward(s, ids, sel): return s.h(ids, sel)


class Packed(nn.Module):
    def __init__(s, h, n_s=None): super().__init__(); s.h = h; s.n_s = n_s
    def forward(s, s_ids, b_ids, sel): return s.h.forward_packed(s_ids, b_ids, sel, n_s=s.n_s)


t0 = time.time()
W, _, _ = hob.load_weights()
h = (hob.HobNL(W) if os.environ.get('HOB_NL') == '1' else hob.Hob(W, dtype=torch.bfloat16, C=a.C, attn='explicit')).eval(); del W
qs = LI['questions']
if a.L:
    tag = a.tag or f'evalL{a.L}'
    mod = Single(h); ex = (torch.zeros(a.L, dtype=torch.long), torch.zeros(a.sel, dtype=torch.long))
elif a.M == 1:
    tag = a.tag or f'T{a.T}_M1'
    q = qs[0]; ids = LI['states'][str(a.T)][0] + q['q']
    r = [a.T + o for o in q['opt']] + [len(ids) - 1]; r += [r[-1]] * (a.sel - len(r))
    tag += f'_S{a.sel}'
    mod = Single(h); ex = (torch.tensor(ids), torch.tensor(r))
else:
    tag = a.tag or f'T{a.T}_M{a.M}'
    qq = qs[:a.M]; Lq = max(len(q['q']) for q in qq); S = max(len(q['opt']) for q in qq) + 1
    Lq = ((Lq + a.C - 1) // a.C) * a.C; Ls = ((a.T + a.C - 1) // a.C) * a.C      # pad to chunk multiples (pads masked / causal-harmless)
    b = torch.zeros(a.M, Lq, dtype=torch.long); sel = torch.zeros(a.M, S, dtype=torch.long)
    for i, q in enumerate(qq):
        b[i, :len(q['q'])] = torch.tensor(q['q']); r = q['opt'] + [len(q['q']) - 1]; r += [r[-1]] * (S - len(r)); sel[i] = torch.tensor(r)
    st = LI['states'][str(a.T)][0] + [0] * (Ls - a.T)
    mod = Packed(h, n_s=a.T if Ls != a.T else None); ex = (torch.tensor(st), b, sel)
tag += f'_C{a.C}' + (f'_{a.cast}' if a.cast != 'none' else '') + (f'_{a.opt}' if a.opt else '')
args = ['--model-type', 'transformer', '--auto-cast', a.cast]
if a.cast != 'none': args += ['--auto-cast-type', 'bf16']
if a.opt: args += [f'-{a.opt}']
print('tracing', tag, [tuple(x.shape) for x in ex], args, 'load %.0fs' % (time.time() - t0), flush=True)
t1 = time.time()
tr = torch_neuronx.trace(mod, ex, compiler_args=args, compiler_workdir=f'{OUT}/wd_{tag}')
print('compiled %.0fs' % (time.time() - t1), flush=True)
torch.jit.save(tr, f'{OUT}/{tag}.pt')
print('saved', tag, flush=True)
