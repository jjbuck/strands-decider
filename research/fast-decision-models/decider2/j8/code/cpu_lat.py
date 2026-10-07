"""cpu_lat.py (SPR AMX box): exact-length latency of the full 24-layer hobson in bf16, 1 and 4 questions, fresh inputs.
  MODE=eager|compile  LENS=64,256,1000,4000  MS=1,4  REPS=12
Writes ~/work/j8/cpu_lat_<tag>.json"""
import os, sys, json, time, statistics as st
sys.path.insert(0, os.path.expanduser('~/work/j8'))
import torch
NT = int(os.environ.get('NT', '16')); torch.set_num_threads(NT)
import hob

MODE = os.environ.get('MODE', 'compile'); TAG = os.environ.get('TAG', MODE)
LENS = [int(x) for x in os.environ.get('LENS', '64,256,1000,4000').split(',')]
MS = [int(x) for x in os.environ.get('MS', '1,4').split(',')]
REPS = int(os.environ.get('REPS', '12'))
C = int(os.environ.get('C', '64'))
LI = json.load(open(os.path.expanduser('~/work/j8/lat_inputs.json')))
W, hs, cfg = hob.load_weights()
model = hob.Hob(W, dtype=torch.bfloat16, C=C, attn='sdpa').eval(); del W
head = hob.Head(hs, cfg)
fwd, fpk = model.forward, model.forward_packed
if MODE == 'compile':
    import torch._inductor.config as ic
    ic.freezing = True
    if os.environ.get('MAXAT'): ic.max_autotune = True; ic.max_autotune_gemm_backends = 'CPP,ATEN'
    torch._dynamo.config.cache_size_limit = 256
    fwd = torch.compile(model.forward, dynamic=False); fpk = torch.compile(model.forward_packed, dynamic=False)
qs = LI['questions']


def inputs(T, M, rep):
    s = LI['states'][str(T)][rep % len(LI['states'][str(T)])]
    if M == 1:
        q = qs[0]; ids = s + q['q']
        return (torch.tensor(ids), torch.tensor([T + o for o in q['opt']] + [len(ids) - 1])), None
    qq = qs[:M]; Lq = max(len(q['q']) for q in qq); S = max(len(q['opt']) for q in qq) + 1
    b = torch.zeros(M, Lq, dtype=torch.long); sel = torch.zeros(M, S, dtype=torch.long)
    for i, q in enumerate(qq):
        b[i, :len(q['q'])] = torch.tensor(q['q']); r = q['opt'] + [len(q['q']) - 1]; r += [r[-1]] * (S - len(r)); sel[i] = torch.tensor(r)
    return (torch.tensor(s), b, sel), qq


def call(T, M, rep):
    x, qq = inputs(T, M, rep)
    with torch.inference_mode():
        if M == 1:
            rows = fwd(*x); return [head.probs(rows, qs[0]['kind'])]
        rows = fpk(*x)
        return [head.probs(rows[i, :len(q['opt']) + 1], q['kind']) for i, q in enumerate(qq)]


res = {}
for T in LENS:
    for M in MS:
        t0 = time.time(); call(T, M, 0); tc = time.time() - t0      # compile / first call
        for r in range(1, 3): call(T, M, r)
        ts = []
        for r in range(3, 3 + REPS):
            t = time.perf_counter(); call(T, M, r); ts.append((time.perf_counter() - t) * 1000)
        ts.sort()
        res[f'T{T}_M{M}'] = dict(median=round(st.median(ts), 2), p95=round(ts[int(0.95 * (len(ts) - 1) + 0.5)], 2), min=round(ts[0], 2), first_s=round(tc, 1), n=len(ts))
        print(T, M, res[f'T{T}_M{M}'], flush=True)
        json.dump(dict(mode=MODE, nt=NT, C=C, res=res), open(os.path.expanduser(f'~/work/j8/cpu_lat_{TAG}.json'), 'w'), indent=1)
# sanity: compiled probabilities vs eager on one 1000-token input
if MODE == 'compile':
    x, _ = inputs(1000, 1, 0)
    with torch.inference_mode():
        a = head.probs(fwd(*x), qs[0]['kind']); b = head.probs(model.forward(*x), qs[0]['kind'])
    print('compile vs eager max|dp|', float((a - b).abs().max()), flush=True)
