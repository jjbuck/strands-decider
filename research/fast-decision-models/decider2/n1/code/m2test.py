"""m2test.py NL K [BLK]: compile M2 truncated to NL layers (k=K) packed T=1000, M=1 -> does it compile, latency."""
import os, sys, json, time, statistics as st
sys.path.insert(0, os.path.expanduser('~/work/n1'))
import torch, torch.nn as nn, torch_neuronx
import hob, m2
NL, K = int(sys.argv[1]), int(sys.argv[2]); BLK = int(sys.argv[3]) if len(sys.argv) > 3 else 256
W, _, _ = hob.load_weights()
h = m2.M2(W, k=K, blk=BLK).eval(); del W
h.L = h.L[:NL]; h.types = h.types[:NL]
h.deep_attn = [i for i in h.deep_attn if i < NL]
class P(nn.Module):
    def __init__(s, h): super().__init__(); s.h = h
    def forward(s, a, b, c): return s.h.forward_packed(a, b, c, n_s=1000)
LI = json.load(open(os.path.expanduser('~/work/n1/lat_inputs.json'))); q = LI['questions'][0]
st_ = torch.tensor(LI['states']['1000'][0] + [0] * 24)
b = torch.zeros(1, 128, dtype=torch.long); b[0, :len(q['q'])] = torch.tensor(q['q'])
sel = torch.tensor([q['opt'] + [len(q['q']) - 1]])
t0 = time.time()
tr = torch_neuronx.trace(P(h), (st_, b, sel), compiler_args=['--model-type', 'transformer', '--auto-cast', 'none'],
                         compiler_workdir=os.path.expanduser(f'~/work/n1/ltest/wd_m2_NL{NL}_K{K}_B{BLK}'), inline_weights_to_neff=False)
torch_neuronx.move_trace_to_device(tr, 0)
for _ in range(3): y = tr(st_, b, sel)
ts = []
for _ in range(10):
    t = time.perf_counter(); y = tr(st_, b, sel); ts.append((time.perf_counter() - t) * 1000)
print(json.dumps(dict(NL=NL, K=K, BLK=BLK, compile_s=round(time.time() - t0), median_ms=round(st.median(ts), 2))), flush=True)
