"""pcheck.py (inf2): packed 4-question graph vs 4 single-question graph calls on the same state (checks the NKI S0 path on HW)."""
import os, sys, json, torch, torch_neuronx
sys.path.insert(0, os.path.expanduser('~/work/n1'))
import hob
from nrun import head
D = os.path.expanduser('~/work/n1'); N = f'{D}/neff'
T = int(sys.argv[1]); ptag, stag, L = sys.argv[2], sys.argv[3], int(sys.argv[4])
LI = json.load(open(f'{D}/lat_inputs.json')); qs = LI['questions']; H = head()
s = LI['states'][str(T)][5]
Ls = ((T + 127) // 128) * 128; Lq = 128
b = torch.zeros(4, Lq, dtype=torch.long); sel = torch.zeros(4, 3, dtype=torch.long)
for i, q in enumerate(qs):
    b[i, :len(q['q'])] = torch.tensor(q['q']); r = q['opt'] + [len(q['q']) - 1]; sel[i] = torch.tensor(r[:3] + [r[-1]] * (3 - len(r)))
mp = torch.jit.load(f'{N}/{ptag}.pt'); rows = mp(torch.tensor(s + [0] * (Ls - T)), b, sel); del mp
ms = torch.jit.load(f'{N}/{stag}.pt')
for i, q in enumerate(qs):
    pp = H.probs(rows[i, :len(q['opt']) + 1], q['kind'])
    ids = s + q['q']; r = [T + o for o in q['opt']] + [len(ids) - 1]
    r1 = ms(torch.tensor(ids + [0] * (L - len(ids))), torch.tensor(r + [r[-1]] * (32 - len(r))))
    ps = H.probs(r1[:len(q['opt']) + 1], q['kind'])
    print(q['qn'], 'packed', [round(x, 4) for x in pp.tolist()], 'single', [round(x, 4) for x in ps.tolist()])
