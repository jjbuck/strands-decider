"""Post-hoc ternarisation of hobson-v19 (the starting point of a BitDistill-style conversion), zero training:
every torso Linear (LoRA merged) -> per-tensor absmean ternary (BitNet's quantiser), activations kept bf16 (W1.58A16) or also int8 per token
(W1.58A8, BitNet's format).  Scored on JB-all + REAL-agree + CF + CF-probe against hobson's references, through plib (merged runtime).
python ptq_hob.py MODE  (MODE = w158a16 | w158a8 | none)  -> results/SUITE.hob_MODE.jsonl"""
import os, sys, json, time
sys.path[:0] = [os.path.expanduser('~/work/tokens'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/sd/src')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch, torch.nn as nn
from kitrun import load_P, load_items, probdict

mode = sys.argv[1]
p = load_P()
n = 0
with torch.no_grad():
    for name, mod in p.tm.named_modules():
        if isinstance(mod, nn.Linear) and mode != 'none':
            W = mod.weight.data.float(); g = W.abs().mean().clamp_min(1e-5)
            mod.weight.data.copy_(((W / g).round().clamp(-1, 1) * g).to(mod.weight.dtype)); n += 1
            if mode == 'w158a8':
                def pre(m, args):
                    x = args[0]; xf = x.float(); s = 127 / xf.abs().amax(-1, keepdim=True).clamp_min(1e-5)
                    return ((xf * s).round().clamp(-128, 127) / s).to(x.dtype),
                mod.register_forward_pre_hook(pre)
print('ternarised linears', n, flush=True)
os.makedirs('results', exist_ok=True)
for su in sys.argv[2:] or ['JB-all', 'REAL-agree', 'CF', 'CF-probe']:
    t0 = time.time(); out = {}
    for it in load_items(su):
        for qn in it['questions']:
            pr = p.prep(it['state'], it['questions'][qn])
            if pr['L'] > 6000: continue
            out.setdefault(it['id'], {})[qn] = probdict(pr['rq'], p.base(pr).tolist())
    with open(f'results/{su}.hob_{mode}.jsonl', 'w') as f:
        for i, q in out.items(): f.write(json.dumps(dict(id=i, q=q)) + '\n')
    print(su, mode, len(out), f'{time.time() - t0:.0f}s', flush=True)
