import os, sys, json, torch
sys.path.insert(0, os.path.expanduser('~/work/n1'))
import hob
it = json.loads(open(os.path.expanduser('~/work/n1/ids.jsonl')).readline())
W, hs, cfg = hob.load_weights(); H = hob.Head(hs, cfg)
ids = it['s'] + it['q']; L = len(ids); Lp = ((L + 127) // 128) * 128
sel = torch.tensor([len(it['s']) + o for o in it['opt']] + [L - 1])
x = torch.tensor(ids + [0] * (Lp - L))
hob.NKI = False; hob.SCAN = True
a = hob.Hob(W, dtype=torch.float32, C=128, attn='explicit').eval()
with torch.inference_mode(): pa = H.probs(a(x, sel), it['kind'])
del a
# HobNL on CPU: replace NKI call by the torch chunk rule for the check
hob.gdn_nki_call = lambda q, k, v, g, b, S0=None: hob.gdn_chunk(q, k, v, g, b, 128, S0)
b = hob.HobNL(W, dtype=torch.float32).eval()
with torch.inference_mode(): pb = H.probs(b(x, sel), it['kind'])
print('Hob', pa.tolist(), 'HobNL', pb.tolist(), 'max|d|', float((pa - pb).abs().max()))
