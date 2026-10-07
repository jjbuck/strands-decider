import os, sys, json, torch
sys.path.insert(0, os.path.expanduser('~/work/j8'))
torch.set_num_threads(int(os.environ.get('NT', '8')))
import hob
LI = json.load(open(os.path.expanduser('~/work/j8/lat_inputs.json'))); qs = LI['questions']
W, hs, cfg = hob.load_weights(); H = hob.Head(hs, cfg)
hob.SCAN = True
hob.gdn_nki_call = lambda q, k, v, g, b, S0=None: hob.gdn_chunk(q, k, v, g, b, 128, S0)
T = 200; s = LI['states']['256'][0][:T]; Ls = 256; Lq = 128
b = torch.zeros(4, Lq, dtype=torch.long); sel = torch.zeros(4, 3, dtype=torch.long)
for i, q in enumerate(qs):
    b[i, :len(q['q'])] = torch.tensor(q['q']); r = q['opt'] + [len(q['q']) - 1]; sel[i] = torch.tensor(r[:3] + [r[-1]] * (3 - len(r)))
st = torch.tensor(s + [0] * (Ls - T))
a = hob.Hob(W, dtype=torch.float32, C=128, attn='explicit').eval()
with torch.inference_mode(): ra = a.forward_packed(st, b, sel, n_s=T)
del a
m = hob.HobNL(W, dtype=torch.float32).eval()
with torch.inference_mode(): rb = m.forward_packed(st, b, sel, n_s=T)
for i, q in enumerate(qs):
    print(q['qn'], H.probs(ra[i, :len(q['opt']) + 1], q['kind']).tolist(), H.probs(rb[i, :len(q['opt']) + 1], q['kind']).tolist())
print('max row diff', float((ra - rb).abs().max()))
