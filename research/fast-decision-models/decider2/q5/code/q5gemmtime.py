"""MEASURED GEMM time of W8A8-b8 vs B12 neuron removal at hobson's shapes (run with exclusive use of the GPU).
int8 GEMMs through torch._int_mm (cuBLASLt s8 x s8 -> s32), bf16 GEMMs through torch.mm, for the 96 GEMMs of one forward at M rows;
B12 shrinks Wgu's N (2 x kept) and Wd's K (kept) per layer, and optionally B13 (layer-23 state rows K/V only; layer-0 Win table).
Fresh random inputs per call, CUDA events, >= 20 warm reps, median and p95 of the whole 96-GEMM sequence.
python q5gemmtime.py --alloc res_alloc.json --M 1125 --Mq 125"""
import os, sys, json, argparse, statistics
import torch
ap = argparse.ArgumentParser(); ap.add_argument('--M', type=int, default=1125); ap.add_argument('--Mq', type=int, default=125)
ap.add_argument('--alloc', default=''); ap.add_argument('--reps', type=int, default=30); ap.add_argument('--out', default=os.path.expanduser('~/work/q5/res_gemmtime.json'))
a = ap.parse_args()
ATT = (3, 7, 11, 15, 19, 23)
B8BF = {(23, 'Wd'), (0, 'Wo'), (7, 'Wo'), (11, 'Wo'), (10, 'Wo'), (23, 'Wo'), (12, 'Wo'), (9, 'Wo')}
dev = 'cuda'


def shapes(keep=None, b13=False, M=1125, Mq=125):
    """list of (rows, N, K, prec) for one forward"""
    out = []
    for i in range(24):
        kp = (keep or {}).get(i, 6144)
        for k in ('Win', 'Wo', 'Wgu', 'Wd'):
            N = {'Win': 5120 if i in ATT else 8224, 'Wo': 2048, 'Wgu': 2 * kp, 'Wd': 2048}[k]
            K = kp if k == 'Wd' else 2048
            p = 'bf16' if (i, k) in B8BF else 'int8'
            rows = M
            if b13 and i == 23:
                if k == 'Win': out.append((M - Mq, 1024, K, p)); rows = Mq       # state rows: K/V only
                else: rows = Mq
            if b13 and i == 0 and k == 'Win': continue                            # per-token table (a gather)
            if N > 0 and K > 0: out.append((rows, N, K, p))
    return out


def make(sh):
    bufs = []
    for (m, n, k, p) in sh:
        m16 = max(32, (m + 7) // 8 * 8); n8 = (n + 7) // 8 * 8; k8 = (k + 7) // 8 * 8          # _int_mm needs M > 16 and multiples of 8
        if p == 'int8':
            bufs.append(('int8', torch.randint(-127, 127, (m16, k8), dtype=torch.int8, device=dev), torch.randint(-127, 127, (n8, k8), dtype=torch.int8, device=dev).t()))
        else:
            bufs.append(('bf16', torch.randn(m16, k8, dtype=torch.bfloat16, device=dev), torch.randn(n8, k8, dtype=torch.bfloat16, device=dev).t()))
    return bufs


def run(bufs):
    for p, x, w in bufs:
        if p == 'int8': torch._int_mm(x, w)
        else: torch.mm(x, w)


def timeit(sh, reps):
    bufs = make(sh); ts = []
    for r in range(reps + 5):
        for j, (p, x, w) in enumerate(bufs):                                          # fresh inputs each call
            if p == 'int8': x.random_(-127, 127)
            else: x.normal_()
        torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); run(bufs); e1.record(); torch.cuda.synchronize()
        if r >= 5: ts.append(e0.elapsed_time(e1))
    ts.sort()
    return dict(median_ms=statistics.median(ts), p95_ms=ts[int(0.95 * (len(ts) - 1))], reps=len(ts))


res = {}
cases = {'b8': (None, False), 'b8+B13': (None, True)}
if a.alloc:
    for nm, kp in json.load(open(a.alloc)).items():
        kp = {int(i): v for i, v in kp.items()}
        cases[nm] = (kp, False); cases[nm + '+B13'] = (kp, True)
for nm, (kp, b13) in cases.items():
    res[nm] = timeit(shapes(kp, b13, a.M, a.Mq), a.reps); print(nm, res[nm], flush=True)
base = res['b8']['median_ms']
for nm in res: res[nm]['ratio_vs_b8'] = res[nm]['median_ms'] / base
json.dump(dict(M=a.M, Mq=a.Mq, res=res, gpu=torch.cuda.get_device_name(0)), open(a.out, 'w'), indent=1)
print({k: round(v['ratio_vs_b8'], 3) for k, v in res.items()})
