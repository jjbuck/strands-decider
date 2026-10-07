import os, sys, json, torch
sys.path.insert(0, os.path.expanduser('~/work/j8'))
torch.set_num_threads(int(os.environ.get('NT', '8')))
import hob
LI = json.load(open(os.path.expanduser('~/work/j8/lat_inputs.json'))); qs = LI['questions']
W, hs, cfg = hob.load_weights(); H = hob.Head(hs, cfg)
m = hob.Hob(W, dtype=torch.float32, C=128, attn='explicit').eval(); del W
T = 200; s = LI['states'][str(256)][0][:T]
Ls = 256; Lq = 128
b = torch.zeros(4, Lq, dtype=torch.long); sel = torch.zeros(4, 3, dtype=torch.long)
for i, q in enumerate(qs):
    b[i, :len(q['q'])] = torch.tensor(q['q']); r = q['opt'] + [len(q['q']) - 1]; sel[i] = torch.tensor(r[:3] + [r[-1]] * (3 - len(r)))
with torch.inference_mode():
    rows = m.forward_packed(torch.tensor(s + [0] * (Ls - T)), b, sel, n_s=T)
    for i, q in enumerate(qs):
        pp = H.probs(rows[i, :len(q['opt']) + 1], q['kind'])
        ids = s + q['q']; L = len(ids); Lp = ((L + 127) // 128) * 128
        r1 = m(torch.tensor(ids + [0] * (Lp - L)), torch.tensor([T + o for o in q['opt']] + [L - 1]))
        ps = H.probs(r1, q['kind'])
        r2 = m(torch.tensor(ids), torch.tensor([T + o for o in q['opt']] + [L - 1]))
        pe = H.probs(r2, q['kind'])
        print(q['qn'], 'packed', [round(x, 5) for x in pp.tolist()], 'single-padded', [round(x, 5) for x in ps.tolist()], 'single-exact', [round(x, 5) for x in pe.tolist()])
